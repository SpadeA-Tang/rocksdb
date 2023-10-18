#include "table/separated_block_based/separated_block_based_table_reader.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "cache/cache_entry_roles.h"
#include "cache/cache_key.h"
#include "cache/sharded_cache.h"
#include "db/compaction/compaction_picker.h"
#include "db/dbformat.h"
#include "db/pinned_iterators_manager.h"
#include "file/file_prefetch_buffer.h"
#include "file/file_util.h"
#include "file/random_access_file_reader.h"
#include "logging/logging.h"
#include "monitoring/perf_context_imp.h"
#include "port/lang.h"
#include "rocksdb/cache.h"
#include "rocksdb/comparator.h"
#include "rocksdb/convenience.h"
#include "rocksdb/env.h"
#include "rocksdb/file_system.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/iterator.h"
#include "rocksdb/options.h"
#include "rocksdb/snapshot.h"
#include "rocksdb/statistics.h"
#include "rocksdb/system_clock.h"
#include "rocksdb/table.h"
#include "rocksdb/table_properties.h"
#include "rocksdb/trace_record.h"
#include "table/block_based/binary_search_index_reader.h"
#include "table/block_based/block.h"
#include "table/block_based/block_based_filter_block.h"
#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/block_based_table_iterator.h"
#include "table/block_based/block_like_traits.h"
#include "table/block_based/block_prefix_index.h"
#include "table/block_based/block_type.h"
#include "table/block_based/filter_block.h"
#include "table/block_based/full_filter_block.h"
#include "table/block_based/hash_index_reader.h"
#include "table/block_based/partitioned_filter_block.h"
#include "table/block_based/partitioned_index_reader.h"
#include "table/block_fetcher.h"
#include "table/format.h"
#include "table/get_context.h"
#include "table/internal_iterator.h"
#include "table/meta_blocks.h"
#include "table/multiget_context.h"
#include "table/persistent_cache_helper.h"
#include "table/persistent_cache_options.h"
#include "table/sst_file_writer_collectors.h"
#include "table/two_level_iterator.h"
#include "test_util/sync_point.h"
#include "util/coding.h"
#include "util/crc32c.h"
#include "util/stop_watch.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {

extern const uint64_t kBlockBasedTableMagicNumber;
extern const std::string kHashIndexPrefixesBlock;
extern const std::string kHashIndexPrefixesMetadataBlock;

SeparatedBlockBasedTable::~SeparatedBlockBasedTable() {
  delete rep_;
}

Status SeparatedBlockBasedTable::Open(
    const ReadOptions& read_options, const ImmutableOptions& ioptions,
    const EnvOptions& env_options, const BlockBasedTableOptions& table_options,
    const InternalKeyComparator& internal_comparator,
    std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
    std::unique_ptr<TableReader>* table_reader,
    const std::shared_ptr<const SliceTransform>& prefix_extractor,
    const bool prefetch_index_and_filter_in_cache, const bool skip_filters,
    const int level, const bool immortal_table,
    const SequenceNumber largest_seqno, const bool force_direct_prefetch,
    TailPrefetchStats* tail_prefetch_stats,
    BlockCacheTracer* const block_cache_tracer,
    size_t max_file_size_for_l0_meta_pin, const std::string& cur_db_session_id,
    uint64_t cur_file_num) {
  table_reader->reset();

  Status s;
  Footer footer;
  std::unique_ptr<FilePrefetchBuffer> prefetch_buffer;

  // Only retain read_options.deadline and read_options.io_timeout.
  // In future, we may retain more
  // options. Specifically, w ignore verify_checksums and default to
  // checksum verification anyway when creating the index and filter
  // readers.
  ReadOptions ro;
  ro.deadline = read_options.deadline;
  ro.io_timeout = read_options.io_timeout;

  // prefetch both index and filters, down to all partitions
  const bool prefetch_all = prefetch_index_and_filter_in_cache || level == 0;
  const bool preload_all = !table_options.cache_index_and_filter_blocks;

  if (!ioptions.allow_mmap_reads) {
    s = PrefetchTail(ro, file.get(), file_size, force_direct_prefetch, tail_prefetch_stats, prefetch_all, preload_all, &prefetch_buffer);
    if (!s.ok()) {
      return s;
    }
  } else {
    // Should not prefetch for mmap mode.
    prefetch_buffer.reset(new FilePrefetchBuffer(
        0 /* readahead_size */, 0 /* max_readahead_size */, false /* enable */,
        true /* track_min_offset */));
  }

  // Read in the following order:
  //    1. Footer
  //    2. [metaindex block]
  //    3. [meta block: properties]
  //    4. [meta block: range deletion tombstone]
  //    5. [meta block: compression dictionary]
  //    6. [meta block: old_index]
  //    7. [meta block: index]
  //    8. [meta block: filter]
  IOOptions opts;
  s = file->PrepareIOOptions(ro, opts);
  if (s.ok()) {
    s = ReadFooterFromFile(opts, file.get(), prefetch_buffer.get(), file_size, &footer, kBlockBasedTableMagicNumber);
  }
  if (!s.ok()) {
    return s;
  }
  if (!IsSupportedFormatVersion(footer.format_version())) {
    return Status::Corruption(
        "Unknown Footer version. Maybe this file was created with newer "
        "version of RocksDB?");
  }

  // We've successfully read the footer. We are ready to serve requests.
  // Better not mutate rep_ after the creation. eg. internal_prefix_transform
  // raw pointer will be used to create HashIndexReader, whose reset may
  // access a dangling pointer.
  BlockCacheLookupContext lookup_context{TableReaderCaller::kPrefetch};
  Rep* rep = new SeparatedBlockBasedTable::Rep(ioptions, env_options, table_options,
                                      internal_comparator, skip_filters,
                                      file_size, level, immortal_table);
  rep->file = std::move(file);
  rep->footer = footer;
  rep->hash_index_allow_collision = table_options.hash_index_allow_collision;
  // We need to wrap data with internal_prefix_transform to make sure it can
  // handle prefix correctly.
  if (prefix_extractor != nullptr) {
    rep->internal_prefix_transform.reset(
        new InternalKeySliceTransform(prefix_extractor.get()));
  }

  // For fully portable/stable cache keys, we need to read the properties
  // block before setting up cache keys. TODO: consider setting up a bootstrap
  // cache key for PersistentCache to use for metaindex and properties blocks.
  rep->persistent_cache_options = PersistentCacheOptions();

  // Meta-blocks are not dictionary compressed. Explicitly set the dictionary
  // handle to null, otherwise it may be seen as uninitialized during the below
  // meta-block reads.
  rep->compression_dict_handle = BlockHandle::NullBlockHandle();

  // Read metaindex
  std::unique_ptr<SeparatedBlockBasedTable> new_table(
      new SeparatedBlockBasedTable(rep, block_cache_tracer));
  std::unique_ptr<Block> metaindex;
  std::unique_ptr<InternalIterator> metaindex_iter;
  s = new_table->ReadMetaIndexBlock(ro, prefetch_buffer.get(), &metaindex,
                                    &metaindex_iter);
  if (!s.ok()) {
    return s;
  }

  // Populates table_properties and some fields that depend on it,
  // such as index_type.
  s = new_table->ReadPropertiesBlock(ro, prefetch_buffer.get(),
                                     metaindex_iter.get(), largest_seqno);
  if (!s.ok()) {
    return s;
  }

  // todo
  if (false) {
    // Establish fast path for unchanged prefix_extractor
    rep->table_prefix_extractor = prefix_extractor;
  } else {
    // Current prefix_extractor doesn't match table
#ifndef ROCKSDB_LITE
    if (rep->table_properties) {
      //**TODO: If/When the DBOptions has a registry in it, the ConfigOptions
      // will need to use it
      ConfigOptions config_options;
      Status st = SliceTransform::CreateFromString(
          config_options, rep->table_properties->prefix_extractor_name,
          &(rep->table_prefix_extractor));
      if (!st.ok()) {
        //**TODO: Should this be error be returned or swallowed?
        ROCKS_LOG_ERROR(rep->ioptions.logger,
                        "Failed to create prefix extractor[%s]: %s",
                        rep->table_properties->prefix_extractor_name.c_str(),
                        st.ToString().c_str());
      }
    }
#endif  // ROCKSDB_LITE
  }

  // With properties loaded, we can set up portable/stable cache keys
  SetupBaseCacheKey(rep->table_properties.get(), cur_db_session_id,
                    cur_file_num, file_size, &rep->base_cache_key);

  rep->persistent_cache_options =
      PersistentCacheOptions(rep->table_options.persistent_cache,
                             rep->base_cache_key, rep->ioptions.stats);

  s = new_table->ReadRangeDelBlock(ro, prefetch_buffer.get(),
                                   metaindex_iter.get(), internal_comparator,
                                   &lookup_context);

  s = new_table->PrefetchIndexAndFilterBlocks(
      ro, prefetch_buffer.get(), metaindex_iter.get(), new_table.get(),
      prefetch_all, table_options, level, file_size,
      max_file_size_for_l0_meta_pin, &lookup_context);

  if (!s.ok()) {
    return s;
  }

  if (s.ok()) {
    // Update tail prefetch stats
    assert(prefetch_buffer.get() != nullptr);
    if (tail_prefetch_stats != nullptr) {
      assert(prefetch_buffer->min_offset_read() < file_size);
      tail_prefetch_stats->RecordEffectiveSize(
          static_cast<size_t>(file_size) - prefetch_buffer->min_offset_read());
    }

    *table_reader = std::move(new_table);
  }

  return Status::OK();
}


Status SeparatedBlockBasedTable::PrefetchTail(
    const ReadOptions& ro, RandomAccessFileReader* file, uint64_t file_size,
    bool force_direct_prefetch, TailPrefetchStats* tail_prefetch_stats,
    const bool prefetch_all, const bool preload_all,
    std::unique_ptr<FilePrefetchBuffer>* prefetch_buffer) {
  size_t tail_prefetch_size = 0;
  if (tail_prefetch_stats != nullptr) {
    // Multiple threads may get a 0 (no history) when running in parallel,
    // but it will get cleared after the first of them finishes.
    tail_prefetch_size = tail_prefetch_stats->GetSuggestedPrefetchSize();
  }
  if (tail_prefetch_size == 0) {
    // Before read footer, readahead backwards to prefetch data. Do more
    // readahead if we're going to read index/filter.
    // TODO: This may incorrectly select small readahead in case partitioned
    // index/filter is enabled and top-level partition pinning is enabled.
    // That's because we need to issue readahead before we read the properties,
    // at which point we don't yet know the index type.
    tail_prefetch_size = prefetch_all || preload_all ? 512 * 1024 : 4 * 1024;
  }
  size_t prefetch_off;
  size_t prefetch_len;
  if (file_size < tail_prefetch_size) {
    prefetch_off = 0;
    prefetch_len = static_cast<size_t>(file_size);
  } else {
    prefetch_off = static_cast<size_t>(file_size - tail_prefetch_size);
    prefetch_len = tail_prefetch_size;
  }

  if (!file->use_direct_io() && !force_direct_prefetch) {
    if (!file->Prefetch(prefetch_off, prefetch_len).IsNotSupported()) {
      prefetch_buffer->reset(new FilePrefetchBuffer(
          0 /* readahead_size */, 0 /* max_readahead_size */,
          false /* enable */, true /* track_min_offset */));
      return Status::OK();
    }
  }

  // Use `FilePrefetchBuffer`
  prefetch_buffer->reset(
      new FilePrefetchBuffer(0 /* readahead_size */, 0 /* max_readahead_size */,
                             true /* enable */, true /* track_min_offset */));
  IOOptions opts;
  Status s = file->PrepareIOOptions(ro, opts);
  if (s.ok()) {
    s = (*prefetch_buffer)->Prefetch(opts, file, prefetch_off, prefetch_len);
  }
  return s;
}

InternalIterator* SeparatedBlockBasedTable::NewIterator(
    const ReadOptions&, const SliceTransform* prefix_extractor, Arena* arena,
    bool skip_filters, TableReaderCaller caller,
    size_t compaction_readahead_size, bool allow_unprepared_value) {
  return nullptr;
}

Status SeparatedBlockBasedTable::Get(const ReadOptions& readOptions,
                                     const Slice& key, GetContext* get_context,
                                     const SliceTransform* prefix_extractor,
                                     bool skip_filters) {
  return Status();
}

uint64_t SeparatedBlockBasedTable::ApproximateOffsetOf(
    const Slice& key, TableReaderCaller caller) {
  return 0;
}

uint64_t SeparatedBlockBasedTable::ApproximateSize(const Slice& start,
                                                   const Slice& end,
                                                   TableReaderCaller caller) {
  return 0;
}

void SeparatedBlockBasedTable::SetupForCompaction() {}
std::shared_ptr<const TableProperties>

SeparatedBlockBasedTable::GetTableProperties() const {
  return std::shared_ptr<const TableProperties>();
}

size_t SeparatedBlockBasedTable::ApproximateMemoryUsage() const { return 0; }

InternalIteratorBase<IndexValue>* SeparatedBlockBasedTable::NewIndexIterator(
    const ReadOptions& read_options, bool need_upper_bound_check,
    IndexBlockIter* input_iter, GetContext* get_context,
    BlockCacheLookupContext* lookup_context) const {
  return nullptr;
}

// Load the meta-index-block from the file. On success, return the loaded
// metaindex
// block and its iterator.
Status SeparatedBlockBasedTable::ReadMetaIndexBlock(
    const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
    std::unique_ptr<Block>* metaindex_block,
    std::unique_ptr<InternalIterator>* iter) {
  return Status::OK();
}

Status SeparatedBlockBasedTable::ReadPropertiesBlock(
    const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
    InternalIterator* meta_iter, const SequenceNumber largest_seqno) {
  return Status::OK();
}

Status SeparatedBlockBasedTable::ReadRangeDelBlock(
    const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
    InternalIterator* meta_iter,
    const InternalKeyComparator& internal_comparator,
    BlockCacheLookupContext* lookup_context) {
  return Status();
}

Status SeparatedBlockBasedTable::PrefetchIndexAndFilterBlocks(
    const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
    InternalIterator* meta_iter, SeparatedBlockBasedTable* new_table, bool prefetch_all,
    const BlockBasedTableOptions& table_options, const int level,
    size_t file_size, size_t max_file_size_for_l0_meta_pin,
    BlockCacheLookupContext* lookup_context) {
  Status s;

  if (rep_->filter_policy) {
    assert(false);
  }
  // Partition filters cannot be enabled without partition indexes
  assert(rep_->filter_type != Rep::FilterType::kPartitionedFilter ||
         rep_->index_type == BlockBasedTableOptions::kTwoLevelIndexSearch);

  // Find compression dictionary handle
  s = FindOptionalMetaBlock(meta_iter, kCompressionDictBlockName,
                            &rep_->compression_dict_handle);
  if (!s.ok()) {
    return s;
  }

  BlockBasedTableOptions::IndexType index_type = rep_->index_type;

  const bool use_cache = table_options.cache_index_and_filter_blocks;

  const bool maybe_flushed =
      level == 0 && file_size <= max_file_size_for_l0_meta_pin;
  std::function<bool(PinningTier, PinningTier)> is_pinned =
      [maybe_flushed, &is_pinned](PinningTier pinning_tier,
                                  PinningTier fallback_pinning_tier) {
        // Fallback to fallback would lead to infinite recursion. Disallow it.
        assert(fallback_pinning_tier != PinningTier::kFallback);

        switch (pinning_tier) {
          case PinningTier::kFallback:
            return is_pinned(fallback_pinning_tier,
                             PinningTier::kNone /* fallback_pinning_tier */);
          case PinningTier::kNone:
            return false;
          case PinningTier::kFlushedAndSimilar:
            return maybe_flushed;
          case PinningTier::kAll:
            return true;
        };

        // In GCC, this is needed to suppress `control reaches end of non-void
        // function [-Werror=return-type]`.
        assert(false);
        return false;
      };
  const bool pin_top_level_index = is_pinned(
      table_options.metadata_cache_options.top_level_index_pinning,
      table_options.pin_top_level_index_and_filter ? PinningTier::kAll
                                                   : PinningTier::kNone);
  const bool pin_partition =
      is_pinned(table_options.metadata_cache_options.partition_pinning,
                table_options.pin_l0_filter_and_index_blocks_in_cache
                    ? PinningTier::kFlushedAndSimilar
                    : PinningTier::kNone);
  const bool pin_unpartitioned =
      is_pinned(table_options.metadata_cache_options.unpartitioned_pinning,
                table_options.pin_l0_filter_and_index_blocks_in_cache
                    ? PinningTier::kFlushedAndSimilar
                    : PinningTier::kNone);

  // pin the first level of index
  const bool pin_index =
      index_type == BlockBasedTableOptions::kTwoLevelIndexSearch
          ? pin_top_level_index
          : pin_unpartitioned;
  // prefetch the first level of index
  // WART: this might be redundant (unnecessary cache hit) if !pin_index,
  // depending on prepopulate_block_cache option
  const bool prefetch_index = prefetch_all || pin_index;

  std::unique_ptr<BlockBasedTable::IndexReader> index_reader;
  s = new_table->CreateIndexReader(ro, prefetch_buffer, meta_iter, use_cache,
                                   prefetch_index, pin_index, lookup_context,
                                   &index_reader);

  return Status();
}

void SeparatedBlockBasedTable::SetupBaseCacheKey(
    const TableProperties* properties, const std::string& cur_db_session_id,
    uint64_t cur_file_number, uint64_t file_size,
    OffsetableCacheKey* out_base_cache_key, bool* out_is_stable) {
  // Use a stable cache key if sufficient data is in table properties
  std::string db_session_id;
  uint64_t file_num;
  std::string db_id;
  if (properties && !properties->db_session_id.empty() &&
      properties->orig_file_number > 0) {
    // (Newer SST file case)
    // We must have both properties to get a stable unique id because
    // CreateColumnFamilyWithImport or IngestExternalFiles can change the
    // file numbers on a file.
    db_session_id = properties->db_session_id;
    file_num = properties->orig_file_number;
    // Less critical, populated in earlier release than above
    db_id = properties->db_id;
    if (out_is_stable) {
      *out_is_stable = true;
    }
  } else {
    // (Old SST file case)
    // We use (unique) cache keys based on current identifiers. These are at
    // least stable across table file close and re-open, but not across
    // different DBs nor DB close and re-open.
    db_session_id = cur_db_session_id;
    file_num = cur_file_number;
    // Plumbing through the DB ID to here would be annoying, and of limited
    // value because of the case of VersionSet::Recover opening some table
    // files and later setting the DB ID. So we just rely on uniqueness
    // level provided by session ID.
    db_id = "unknown";
    if (out_is_stable) {
      *out_is_stable = false;
    }

    // Minimum block size is 5 bytes; therefore we can trim off two lower bits
    // from offets. See GetCacheKey.
    *out_base_cache_key = OffsetableCacheKey(db_id, db_session_id, file_num,
                                             /*max_offset*/ file_size >> 2);
  }
}

// REQUIRES: The following fields of rep_ should have already been populated:
//  1. file
//  2. index_handle,
//  3. options
//  4. internal_comparator
//  5. index_type
Status SeparatedBlockBasedTable::CreateIndexReader(
    const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
    InternalIterator* preloaded_meta_index_iter, bool use_cache, bool prefetch,
    bool pin, BlockCacheLookupContext* lookup_context,
    std::unique_ptr<BlockBasedTable::IndexReader>* index_reader, std::unique_ptr<BlockBasedTable::IndexReader>* old_index_reader) {
  // kHashSearch requires non-empty prefix_extractor but bypass checking
  // prefix_extractor here since we have no access to MutableCFOptions.
  // Add need_upper_bound_check flag in  BlockBasedTable::NewIndexIterator.
  // If prefix_extractor does not match prefix_extractor_name from table
  // properties, turn off Hash Index by setting total_order_seek to true

  switch (rep_->index_type) {
    case BlockBasedTableOptions::kBinarySearch:
      FALLTHROUGH_INTENDED;
    case BlockBasedTableOptions::kBinarySearchWithFirstKey: {
      return BinarySearchIndexReader::Create(this, ro, prefetch_buffer,
                                             use_cache, prefetch, pin,
                                             lookup_context, index_reader);
    }
    default: {
      // unimplemented
      assert(false);
    }
  }
  return Status();
}

}