#include "table/separated_block_based/separated_block_based_table_factory.h"

#include <stdint.h>

#include <cinttypes>
#include <memory>
#include <string>

#include "cache/cache_entry_roles.h"
#include "logging/logging.h"
#include "options/options_helper.h"
#include "port/port.h"
#include "rocksdb/cache.h"
#include "rocksdb/convenience.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/flush_block_policy.h"
#include "rocksdb/rocksdb_namespace.h"
#include "rocksdb/table.h"
#include "rocksdb/utilities/options_type.h"
#include "separated_block_based_table_builder.h"
#include "table/block_based/block_based_table_builder.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/format.h"
#include "test_util/testutil.h"
#include "util/mutexlock.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {

#ifndef ROCKSDB_LITE

const std::string kOptNameMetadataCacheOpts = "metadata_cache_options";

static std::unordered_map<std::string, PinningTier>
    pinning_tier_type_string_map = {
        {"kFallback", PinningTier::kFallback},
        {"kNone", PinningTier::kNone},
        {"kFlushedAndSimilar", PinningTier::kFlushedAndSimilar},
        {"kAll", PinningTier::kAll}};

static std::unordered_map<std::string, BlockBasedTableOptions::IndexType>
    block_base_table_index_type_string_map = {
        {"kBinarySearch", BlockBasedTableOptions::IndexType::kBinarySearch},
        {"kHashSearch", BlockBasedTableOptions::IndexType::kHashSearch},
        {"kTwoLevelIndexSearch",
         BlockBasedTableOptions::IndexType::kTwoLevelIndexSearch},
        {"kBinarySearchWithFirstKey",
         BlockBasedTableOptions::IndexType::kBinarySearchWithFirstKey}};

static std::unordered_map<std::string,
                          BlockBasedTableOptions::DataBlockIndexType>
    block_base_table_data_block_index_type_string_map = {
        {"kDataBlockBinarySearch",
         BlockBasedTableOptions::DataBlockIndexType::kDataBlockBinarySearch},
        {"kDataBlockBinaryAndHash",
         BlockBasedTableOptions::DataBlockIndexType::kDataBlockBinaryAndHash}};

static std::unordered_map<std::string,
                          BlockBasedTableOptions::IndexShorteningMode>
    block_base_table_index_shortening_mode_string_map = {
        {"kNoShortening",
         BlockBasedTableOptions::IndexShorteningMode::kNoShortening},
        {"kShortenSeparators",
         BlockBasedTableOptions::IndexShorteningMode::kShortenSeparators},
        {"kShortenSeparatorsAndSuccessor",
         BlockBasedTableOptions::IndexShorteningMode::
             kShortenSeparatorsAndSuccessor}};

static std::unordered_map<std::string, OptionTypeInfo>
    metadata_cache_options_type_info = {
        {"top_level_index_pinning",
         OptionTypeInfo::Enum<PinningTier>(
             offsetof(struct MetadataCacheOptions, top_level_index_pinning),
             &pinning_tier_type_string_map)},
        {"partition_pinning",
         OptionTypeInfo::Enum<PinningTier>(
             offsetof(struct MetadataCacheOptions, partition_pinning),
             &pinning_tier_type_string_map)},
        {"unpartitioned_pinning",
         OptionTypeInfo::Enum<PinningTier>(
             offsetof(struct MetadataCacheOptions, unpartitioned_pinning),
             &pinning_tier_type_string_map)}};

static std::unordered_map<std::string,
                          BlockBasedTableOptions::PrepopulateBlockCache>
    block_base_table_prepopulate_block_cache_string_map = {
        {"kDisable", BlockBasedTableOptions::PrepopulateBlockCache::kDisable},
        {"kFlushOnly",
         BlockBasedTableOptions::PrepopulateBlockCache::kFlushOnly}};

#endif  // ROCKSDB_LITE

static std::unordered_map<std::string, OptionTypeInfo>
    block_based_table_type_info = {
#ifndef ROCKSDB_LITE
        /* currently not supported
          std::shared_ptr<Cache> block_cache = nullptr;
          std::shared_ptr<Cache> block_cache_compressed = nullptr;
         */
        {"flush_block_policy_factory",
         OptionTypeInfo::AsCustomSharedPtr<FlushBlockPolicyFactory>(
             offsetof(struct BlockBasedTableOptions,
                      flush_block_policy_factory),
             OptionVerificationType::kByName, OptionTypeFlags::kCompareNever)},
        {"cache_index_and_filter_blocks",
         {offsetof(struct BlockBasedTableOptions,
                   cache_index_and_filter_blocks),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"cache_index_and_filter_blocks_with_high_priority",
         {offsetof(struct BlockBasedTableOptions,
                   cache_index_and_filter_blocks_with_high_priority),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"pin_l0_filter_and_index_blocks_in_cache",
         {offsetof(struct BlockBasedTableOptions,
                   pin_l0_filter_and_index_blocks_in_cache),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"index_type", OptionTypeInfo::Enum<BlockBasedTableOptions::IndexType>(
                           offsetof(struct BlockBasedTableOptions, index_type),
                           &block_base_table_index_type_string_map)},
        {"hash_index_allow_collision",
         {offsetof(struct BlockBasedTableOptions, hash_index_allow_collision),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"data_block_index_type",
         OptionTypeInfo::Enum<BlockBasedTableOptions::DataBlockIndexType>(
             offsetof(struct BlockBasedTableOptions, data_block_index_type),
             &block_base_table_data_block_index_type_string_map)},
        {"index_shortening",
         OptionTypeInfo::Enum<BlockBasedTableOptions::IndexShorteningMode>(
             offsetof(struct BlockBasedTableOptions, index_shortening),
             &block_base_table_index_shortening_mode_string_map)},
        {"data_block_hash_table_util_ratio",
         {offsetof(struct BlockBasedTableOptions,
                   data_block_hash_table_util_ratio),
          OptionType::kDouble, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"checksum",
         {offsetof(struct BlockBasedTableOptions, checksum),
          OptionType::kChecksumType, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"no_block_cache",
         {offsetof(struct BlockBasedTableOptions, no_block_cache),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"block_size",
         {offsetof(struct BlockBasedTableOptions, block_size),
          OptionType::kSizeT, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"block_size_deviation",
         {offsetof(struct BlockBasedTableOptions, block_size_deviation),
          OptionType::kInt, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"block_restart_interval",
         {offsetof(struct BlockBasedTableOptions, block_restart_interval),
          OptionType::kInt, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"index_block_restart_interval",
         {offsetof(struct BlockBasedTableOptions, index_block_restart_interval),
          OptionType::kInt, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"index_per_partition",
         {0, OptionType::kUInt64T, OptionVerificationType::kDeprecated,
          OptionTypeFlags::kNone}},
        {"metadata_block_size",
         {offsetof(struct BlockBasedTableOptions, metadata_block_size),
          OptionType::kUInt64T, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"partition_filters",
         {offsetof(struct BlockBasedTableOptions, partition_filters),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"optimize_filters_for_memory",
         {offsetof(struct BlockBasedTableOptions, optimize_filters_for_memory),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"filter_policy",
         {offsetof(struct BlockBasedTableOptions, filter_policy),
          OptionType::kUnknown, OptionVerificationType::kByNameAllowFromNull,
          OptionTypeFlags::kNone,
          // Parses the Filter policy
          [](const ConfigOptions& opts, const std::string&,
             const std::string& value, void* addr) {
            auto* policy =
                static_cast<std::shared_ptr<const FilterPolicy>*>(addr);
            return FilterPolicy::CreateFromString(opts, value, policy);
          },
          // Converts the FilterPolicy to its string representation
          [](const ConfigOptions&, const std::string&, const void* addr,
             std::string* value) {
            const auto* policy =
                static_cast<const std::shared_ptr<const FilterPolicy>*>(addr);
            if (policy->get()) {
              *value = (*policy)->Name();
            } else {
              *value = kNullptrString;
            }
            return Status::OK();
          },
          // Compares two FilterPolicy objects for equality
          [](const ConfigOptions&, const std::string&, const void* addr1,
             const void* addr2, std::string*) {
            const auto* policy1 =
                static_cast<const std::shared_ptr<const FilterPolicy>*>(addr1)
                    ->get();
            const auto* policy2 =
                static_cast<const std::shared_ptr<FilterPolicy>*>(addr2)->get();
            if (policy1 == policy2) {
              return true;
            } else if (policy1 != nullptr && policy2 != nullptr) {
              return (strcmp(policy1->Name(), policy2->Name()) == 0);
            } else {
              return false;
            }
          }}},
        {"whole_key_filtering",
         {offsetof(struct BlockBasedTableOptions, whole_key_filtering),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"reserve_table_builder_memory",
         {offsetof(struct BlockBasedTableOptions, reserve_table_builder_memory),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"skip_table_builder_flush",
         {0, OptionType::kBoolean, OptionVerificationType::kDeprecated,
          OptionTypeFlags::kNone}},
        {"format_version",
         {offsetof(struct BlockBasedTableOptions, format_version),
          OptionType::kUInt32T, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"verify_compression",
         {offsetof(struct BlockBasedTableOptions, verify_compression),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"read_amp_bytes_per_bit",
         {offsetof(struct BlockBasedTableOptions, read_amp_bytes_per_bit),
          OptionType::kUInt32T, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone,
          [](const ConfigOptions& /*opts*/, const std::string& /*name*/,
             const std::string& value, void* addr) {
            // A workaround to fix a bug in 6.10, 6.11, 6.12, 6.13
            // and 6.14. The bug will write out 8 bytes to OPTIONS file from the
            // starting address of BlockBasedTableOptions.read_amp_bytes_per_bit
            // which is actually a uint32. Consequently, the value of
            // read_amp_bytes_per_bit written in the OPTIONS file is wrong.
            // From 6.15, RocksDB will try to parse the read_amp_bytes_per_bit
            // from OPTIONS file as a uint32. To be able to load OPTIONS file
            // generated by affected releases before the fix, we need to
            // manually parse read_amp_bytes_per_bit with this special hack.
            uint64_t read_amp_bytes_per_bit = ParseUint64(value);
            *(static_cast<uint32_t*>(addr)) =
                static_cast<uint32_t>(read_amp_bytes_per_bit);
            return Status::OK();
          }}},
        {"enable_index_compression",
         {offsetof(struct BlockBasedTableOptions, enable_index_compression),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"block_align",
         {offsetof(struct BlockBasedTableOptions, block_align),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {"pin_top_level_index_and_filter",
         {offsetof(struct BlockBasedTableOptions,
                   pin_top_level_index_and_filter),
          OptionType::kBoolean, OptionVerificationType::kNormal,
          OptionTypeFlags::kNone}},
        {kOptNameMetadataCacheOpts,
         OptionTypeInfo::Struct(
             kOptNameMetadataCacheOpts, &metadata_cache_options_type_info,
             offsetof(struct BlockBasedTableOptions, metadata_cache_options),
             OptionVerificationType::kNormal, OptionTypeFlags::kNone)},
        {"block_cache",
         {offsetof(struct BlockBasedTableOptions, block_cache),
          OptionType::kUnknown, OptionVerificationType::kNormal,
          (OptionTypeFlags::kCompareNever | OptionTypeFlags::kDontSerialize),
          // Parses the input vsalue as a Cache
          [](const ConfigOptions& opts, const std::string&,
             const std::string& value, void* addr) {
            auto* cache = static_cast<std::shared_ptr<Cache>*>(addr);
            return Cache::CreateFromString(opts, value, cache);
          }}},
        {"block_cache_compressed",
         {offsetof(struct BlockBasedTableOptions, block_cache_compressed),
          OptionType::kUnknown, OptionVerificationType::kNormal,
          (OptionTypeFlags::kCompareNever | OptionTypeFlags::kDontSerialize),
          // Parses the input vsalue as a Cache
          [](const ConfigOptions& opts, const std::string&,
             const std::string& value, void* addr) {
            auto* cache = static_cast<std::shared_ptr<Cache>*>(addr);
            return Cache::CreateFromString(opts, value, cache);
          }}},
        {"max_auto_readahead_size",
         {offsetof(struct BlockBasedTableOptions, max_auto_readahead_size),
          OptionType::kSizeT, OptionVerificationType::kNormal,
          OptionTypeFlags::kMutable}},
        {"prepopulate_block_cache",
         OptionTypeInfo::Enum<BlockBasedTableOptions::PrepopulateBlockCache>(
             offsetof(struct BlockBasedTableOptions, prepopulate_block_cache),
             &block_base_table_prepopulate_block_cache_string_map,
             OptionTypeFlags::kMutable)},

#endif  // ROCKSDB_LITE
};

SeparatedBlockBasedTableFactory::SeparatedBlockBasedTableFactory(
    const BlockBasedTableOptions& _table_options)
    : table_options_(_table_options) {
  InitializeOptions();
  RegisterOptions(&table_options_, &block_based_table_type_info);
}

void SeparatedBlockBasedTableFactory::InitializeOptions() {
  if (table_options_.flush_block_policy_factory == nullptr) {
    table_options_.flush_block_policy_factory.reset(
        new FlushBlockBySizePolicyFactory());
  }
  if (table_options_.no_block_cache) {
    table_options_.block_cache.reset();
  } else if (table_options_.block_cache == nullptr) {
    LRUCacheOptions co;
    co.capacity = 8 << 20;
    // It makes little sense to pay overhead for mid-point insertion while the
    // block size is only 8MB.
    co.high_pri_pool_ratio = 0.0;
    table_options_.block_cache = NewLRUCache(co);
  }
  if (table_options_.block_restart_interval < 1) {
    table_options_.block_restart_interval = 1;
  }
  if (table_options_.index_block_restart_interval < 1) {
    table_options_.index_block_restart_interval = 1;
  }
  if (table_options_.index_type == BlockBasedTableOptions::kHashSearch &&
      table_options_.index_block_restart_interval != 1) {
    // Currently kHashSearch is incompatible with index_block_restart_interval >
    // 1
    table_options_.index_block_restart_interval = 1;
  }
  if (table_options_.partition_filters &&
      table_options_.index_type !=
          BlockBasedTableOptions::kTwoLevelIndexSearch) {
    // We do not support partitioned filters without partitioning indexes
    table_options_.partition_filters = false;
  }
}

TableBuilder* SeparatedBlockBasedTableFactory::NewTableBuilder(
    const TableBuilderOptions& table_builder_options,
    WritableFileWriter* file) const {
  return new SeparatedBlockBasedTableBuilder(
      table_options_, table_builder_options,
      *test::BytewiseComparatorWithU64TsWrapper(), file);
}

Status SeparatedBlockBasedTableFactory::NewTableReader(
    const ReadOptions& ro, const TableReaderOptions& table_reader_options,
    std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
    std::unique_ptr<TableReader>* table_reader,
    bool prefetch_index_and_filter_in_cache) const {
  return Status{};
}

TableFactory* NewSeparatedBlockBasedTableFactory(
    const BlockBasedTableOptions& _table_options) {
  return new SeparatedBlockBasedTableFactory(_table_options);
}

}  // namespace ROCKSDB_NAMESPACE