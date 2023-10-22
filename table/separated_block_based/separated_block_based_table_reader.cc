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
#include "separated_block_based_table_iterator.h"
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
#include "table/separated_block_based/binary_search_index_reader.h"
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

namespace {
// Read the block identified by "handle" from "file".
// The only relevant option is options.verify_checksums for now.
// On failure return non-OK.
// On success fill *result and return OK - caller owns *result
// @param uncompression_dict Data for presetting the compression library's
//    dictionary.
template <typename TBlocklike>
Status ReadBlockFromFile(
    RandomAccessFileReader* file, FilePrefetchBuffer* prefetch_buffer,
    const Footer& footer, const ReadOptions& options, const BlockHandle& handle,
    std::unique_ptr<TBlocklike>* result, const ImmutableOptions& ioptions,
    bool do_uncompress, bool maybe_compressed, BlockType block_type,
    const UncompressionDict& uncompression_dict,
    const PersistentCacheOptions& cache_options, size_t read_amp_bytes_per_bit,
    MemoryAllocator* memory_allocator, bool for_compaction, bool using_zstd,
    const FilterPolicy* filter_policy) {
  assert(result);

  BlockContents contents;
  BlockFetcher block_fetcher(
      file, prefetch_buffer, footer, options, handle, &contents, ioptions,
      do_uncompress, maybe_compressed, block_type, uncompression_dict,
      cache_options, memory_allocator, nullptr, for_compaction);
  Status s = block_fetcher.ReadBlockContents();
  if (s.ok()) {
    result->reset(BlocklikeTraits<TBlocklike>::Create(
        std::move(contents), read_amp_bytes_per_bit, ioptions.stats, using_zstd,
        filter_policy));
  }

  return s;
}

}  // namespace

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

bool SeparatedBlockBasedTable::PrefixExtractorChanged(
    const SliceTransform* prefix_extractor) const {
  if (prefix_extractor == nullptr) {
    return true;
  } else if (prefix_extractor == rep_->table_prefix_extractor.get()) {
    return false;
  } else {
    assert(false);
    //    return PrefixExtractorChangedHelper(rep_->table_properties.get(),
    //                                        prefix_extractor);
  }
}

InternalIterator* SeparatedBlockBasedTable::NewIterator(
    const ReadOptions& read_options, const SliceTransform* prefix_extractor,
    Arena* arena, bool skip_filters, TableReaderCaller caller,
    size_t compaction_readahead_size, bool allow_unprepared_value) {
  BlockCacheLookupContext lookup_context{caller};
  bool need_upper_bound_check =
      read_options.auto_prefix_mode || PrefixExtractorChanged(prefix_extractor);
  std::unique_ptr<InternalIteratorBase<IndexValue>> index_iter(NewIndexIterator(
      read_options,
      need_upper_bound_check &&
          rep_->index_type == BlockBasedTableOptions::kHashSearch,
      nullptr, nullptr, &lookup_context, false));

  std::unique_ptr<InternalIteratorBase<IndexValue>> old_index_iter(
      NewIndexIterator(
          read_options,
          need_upper_bound_check &&
              rep_->index_type == BlockBasedTableOptions::kHashSearch,
          nullptr, nullptr, &lookup_context, true));

  if (arena == nullptr) {
    return new SeparatedBlockBasedTableIterator(
        this, read_options, rep_->internal_comparator, std::move(index_iter),
        std::move(old_index_iter),
        !skip_filters && !read_options.total_order_seek &&
            prefix_extractor != nullptr,
        need_upper_bound_check, prefix_extractor, caller,
        compaction_readahead_size, allow_unprepared_value);
  } else {
  }

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
    const ReadOptions& read_options, bool disable_prefix_seek,
    IndexBlockIter* input_iter, GetContext* get_context,
    BlockCacheLookupContext* lookup_context, bool old_data_block) const {
  assert(rep_ != nullptr);
  assert(rep_->index_reader != nullptr);

  if (!old_data_block) {
    return rep_->index_reader->NewIterator(read_options, disable_prefix_seek,
                                           input_iter, get_context,
                                           lookup_context);
  } else {
    return rep_->old_index_reader->NewIterator(read_options,
                                               disable_prefix_seek, input_iter,
                                               get_context, lookup_context);
  }
}

// Load the meta-index-block from the file. On success, return the loaded
// metaindex
// block and its iterator.
Status SeparatedBlockBasedTable::ReadMetaIndexBlock(
    const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
    std::unique_ptr<Block>* metaindex_block,
    std::unique_ptr<InternalIterator>* iter) {
  std::unique_ptr<Block> metaindex;
  Status s = ReadBlockFromFile(
      rep_->file.get(), prefetch_buffer, rep_->footer, ro,
      rep_->footer.metaindex_handle(), &metaindex, rep_->ioptions,
      true /* decompress */, true /*maybe_compressed*/, BlockType::kMetaIndex,
      UncompressionDict::GetEmptyDict(), rep_->persistent_cache_options,
      0 /* read_amp_bytes_per_bit */, GetMemoryAllocator(rep_->table_options),
      false /* for_compaction */, rep_->blocks_definitely_zstd_compressed,
      nullptr /* filter_policy */);

  if (!s.ok()) {
    ROCKS_LOG_ERROR(rep_->ioptions.logger,
                    "Encountered error while reading data from properties"
                    " block %s",
                    s.ToString().c_str());
    return s;
  }

  *metaindex_block = std::move(metaindex);
  // meta block uses bytewise comparator.
  iter->reset(metaindex_block->get()->NewMetaIterator());
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
  Status s;
  BlockHandle handle;
  s = FindOptionalMetaBlock(meta_iter, kPropertiesBlockName, &handle);

  if (!s.ok()) {
    ROCKS_LOG_WARN(rep_->ioptions.logger,
                   "Error when seeking to properties block from file: %s",
                   s.ToString().c_str());
  } else if (!handle.IsNull()) {
    s = meta_iter->status();
    std::unique_ptr<TableProperties> table_properties;
    if (s.ok()) {
      s = ReadTablePropertiesHelper(
          ro, handle, rep_->file.get(), prefetch_buffer, rep_->footer,
          rep_->ioptions, &table_properties, nullptr /* memory_allocator */);
    }
    IGNORE_STATUS_IF_ERROR(s);

    if (!s.ok()) {
      ROCKS_LOG_WARN(rep_->ioptions.logger,
                     "Encountered error while reading data from properties "
                     "block %s",
                     s.ToString().c_str());
    } else {
      assert(table_properties != nullptr);
      rep_->table_properties = std::move(table_properties);
      rep_->blocks_maybe_compressed =
          rep_->table_properties->compression_name !=
          CompressionTypeToString(kNoCompression);
      rep_->blocks_definitely_zstd_compressed =
          (rep_->table_properties->compression_name ==
               CompressionTypeToString(kZSTD) ||
           rep_->table_properties->compression_name ==
               CompressionTypeToString(kZSTDNotFinalCompression));
    }
  } else {
    ROCKS_LOG_ERROR(rep_->ioptions.logger,
                    "Cannot find Properties block from file.");
  }

  // Read the table properties, if provided.
  if (rep_->table_properties) {
    // todo
    rep_->index_key_includes_seq =
        rep_->table_properties->index_key_is_user_key == 0;
    rep_->index_value_is_full =
        rep_->table_properties->index_value_is_delta_encoded == 0;
    // todo
  }

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
  std::unique_ptr<BlockBasedTable::IndexReader> old_index_reader;
  s = new_table->CreateIndexReader(ro, prefetch_buffer, meta_iter, use_cache,
                                   prefetch_index, pin_index, lookup_context,
                                   &index_reader, &old_index_reader);
  if (!s.ok()) {
    return s;
  }

  rep_->index_reader = std::move(index_reader);
  rep_->old_index_reader = std::move(old_index_reader);

  // The partitions of partitioned index are always stored in cache. They
  // are hence follow the configuration for pin and prefetch regardless of
  // the value of cache_index_and_filter_blocks
  if (prefetch_all || pin_partition) {
    s = rep_->index_reader->CacheDependencies(ro, pin_partition);
    s = rep_->old_index_reader->CacheDependencies(ro, pin_partition);
  }
  if (!s.ok()) {
    return s;
  }

  // pin the first level of filter
  const bool pin_filter =
      rep_->filter_type == Rep::FilterType::kPartitionedFilter
          ? pin_top_level_index
          : pin_unpartitioned;
  // prefetch the first level of filter
  // WART: this might be redundant (unnecessary cache hit) if !pin_filter,
  // depending on prepopulate_block_cache option
  const bool prefetch_filter = prefetch_all || pin_filter;

  if (rep_->filter_policy) {
    assert(false);
  }

  if (!rep_->compression_dict_handle.IsNull()) {
    assert(false);
  }

  assert(s.ok());
  return s;
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
      return SeparatedBinarySearchIndexReader::Create(
          this, ro, prefetch_buffer, use_cache, prefetch, pin, lookup_context,
          index_reader, old_index_reader);
    }
    default: {
      // unimplemented
      assert(false);
    }
  }

  return Status();
}

CacheKey SeparatedBlockBasedTable::GetCacheKey(
    const OffsetableCacheKey& base_cache_key, const BlockHandle& handle) {
  // Minimum block size is 5 bytes; therefore we can trim off two lower bits
  // from offet.
  return base_cache_key.WithOffset(handle.offset() >> 2);
}

void SeparatedBlockBasedTable::UpdateCacheInsertionMetrics(
    BlockType block_type, GetContext* get_context, size_t usage, bool redundant,
    Statistics* const statistics) {
  // TODO: introduce perf counters for block cache insertions
  if (get_context) {
    ++get_context->get_context_stats_.num_cache_add;
    if (redundant) {
      ++get_context->get_context_stats_.num_cache_add_redundant;
    }
    get_context->get_context_stats_.num_cache_bytes_write += usage;
  } else {
    RecordTick(statistics, BLOCK_CACHE_ADD);
    if (redundant) {
      RecordTick(statistics, BLOCK_CACHE_ADD_REDUNDANT);
    }
    RecordTick(statistics, BLOCK_CACHE_BYTES_WRITE, usage);
  }

  switch (block_type) {
    case BlockType::kFilter:
      if (get_context) {
        ++get_context->get_context_stats_.num_cache_filter_add;
        if (redundant) {
          ++get_context->get_context_stats_.num_cache_filter_add_redundant;
        }
        get_context->get_context_stats_.num_cache_filter_bytes_insert += usage;
      } else {
        RecordTick(statistics, BLOCK_CACHE_FILTER_ADD);
        if (redundant) {
          RecordTick(statistics, BLOCK_CACHE_FILTER_ADD_REDUNDANT);
        }
        RecordTick(statistics, BLOCK_CACHE_FILTER_BYTES_INSERT, usage);
      }
      break;

    case BlockType::kCompressionDictionary:
      if (get_context) {
        ++get_context->get_context_stats_.num_cache_compression_dict_add;
        if (redundant) {
          ++get_context->get_context_stats_
                .num_cache_compression_dict_add_redundant;
        }
        get_context->get_context_stats_
            .num_cache_compression_dict_bytes_insert += usage;
      } else {
        RecordTick(statistics, BLOCK_CACHE_COMPRESSION_DICT_ADD);
        if (redundant) {
          RecordTick(statistics, BLOCK_CACHE_COMPRESSION_DICT_ADD_REDUNDANT);
        }
        RecordTick(statistics, BLOCK_CACHE_COMPRESSION_DICT_BYTES_INSERT,
                   usage);
      }
      break;

    case BlockType::kIndex:
      if (get_context) {
        ++get_context->get_context_stats_.num_cache_index_add;
        if (redundant) {
          ++get_context->get_context_stats_.num_cache_index_add_redundant;
        }
        get_context->get_context_stats_.num_cache_index_bytes_insert += usage;
      } else {
        RecordTick(statistics, BLOCK_CACHE_INDEX_ADD);
        if (redundant) {
          RecordTick(statistics, BLOCK_CACHE_INDEX_ADD_REDUNDANT);
        }
        RecordTick(statistics, BLOCK_CACHE_INDEX_BYTES_INSERT, usage);
      }
      break;

    default:
      // TODO: introduce dedicated tickers/statistics/counters
      // for range tombstones
      if (get_context) {
        ++get_context->get_context_stats_.num_cache_data_add;
        if (redundant) {
          ++get_context->get_context_stats_.num_cache_data_add_redundant;
        }
        get_context->get_context_stats_.num_cache_data_bytes_insert += usage;
      } else {
        RecordTick(statistics, BLOCK_CACHE_DATA_ADD);
        if (redundant) {
          RecordTick(statistics, BLOCK_CACHE_DATA_ADD_REDUNDANT);
        }
        RecordTick(statistics, BLOCK_CACHE_DATA_BYTES_INSERT, usage);
      }
      break;
  }
}

template <>
DataBlockIter* SeparatedBlockBasedTable::InitBlockIterator<DataBlockIter>(
    const Rep* rep, Block* block, BlockType block_type,
    DataBlockIter* input_iter, bool block_contents_pinned) {
  return block->NewDataIterator(rep->internal_comparator.user_comparator(),
                                rep->get_global_seqno(block_type), input_iter,
                                rep->ioptions.stats, block_contents_pinned);
}

template <>
IndexBlockIter* SeparatedBlockBasedTable::InitBlockIterator<IndexBlockIter>(
    const Rep* rep, Block* block, BlockType block_type,
    IndexBlockIter* input_iter, bool block_contents_pinned) {
  return block->NewIndexIterator(
      rep->internal_comparator.user_comparator(),
      rep->get_global_seqno(block_type), input_iter, rep->ioptions.stats,
      /* total_order_seek */ true, rep->index_has_first_key,
      rep->index_key_includes_seq, rep->index_value_is_full,
      block_contents_pinned);
}

Cache::Handle* SeparatedBlockBasedTable::GetEntryFromCache(
    const CacheTier& cache_tier, Cache* block_cache, const Slice& key,
    BlockType block_type, const bool wait, GetContext* get_context,
    const Cache::CacheItemHelper* cache_helper,
    const Cache::CreateCallback& create_cb, Cache::Priority priority) const {
  Cache::Handle* cache_handle = nullptr;
  if (cache_tier == CacheTier::kNonVolatileBlockTier) {
    cache_handle = block_cache->Lookup(key, cache_helper, create_cb, priority,
                                       wait, rep_->ioptions.statistics.get());
  } else {
    cache_handle = block_cache->Lookup(key, rep_->ioptions.statistics.get());
  }

  if (cache_handle != nullptr) {
    //    UpdateCacheHitMetrics(block_type, get_context,
    //                          block_cache->GetUsage(cache_handle));
  } else {
    //    UpdateCacheMissMetrics(block_type, get_context);
  }

  return cache_handle;
}

template <typename TBlocklike>
Status SeparatedBlockBasedTable::GetDataBlockFromCache(
    const Slice& cache_key, Cache* block_cache, Cache* block_cache_compressed,
    const ReadOptions& read_options, CachableEntry<TBlocklike>* block,
    const UncompressionDict& uncompression_dict, BlockType block_type,
    const bool wait, GetContext* get_context) const {
  const size_t read_amp_bytes_per_bit =
      block_type == BlockType::kData
          ? rep_->table_options.read_amp_bytes_per_bit
          : 0;
  assert(block);
  assert(block->IsEmpty());
  const Cache::Priority priority =
      rep_->table_options.cache_index_and_filter_blocks_with_high_priority &&
              (block_type == BlockType::kFilter ||
               block_type == BlockType::kCompressionDictionary ||
               block_type == BlockType::kIndex)
          ? Cache::Priority::HIGH
          : Cache::Priority::LOW;

  Status s;
  BlockContents* compressed_block = nullptr;
  Cache::Handle* block_cache_compressed_handle = nullptr;
  Statistics* statistics = rep_->ioptions.statistics.get();
  bool using_zstd = rep_->blocks_definitely_zstd_compressed;
  const FilterPolicy* filter_policy = rep_->filter_policy;
  Cache::CreateCallback create_cb = GetCreateCallback<TBlocklike>(
      read_amp_bytes_per_bit, statistics, using_zstd, filter_policy);

  // Lookup uncompressed cache first
  if (block_cache != nullptr) {
    assert(!cache_key.empty());
    Cache::Handle* cache_handle = nullptr;
    cache_handle = GetEntryFromCache(
        rep_->ioptions.lowest_used_cache_tier, block_cache, cache_key,
        block_type, wait, get_context,
        BlocklikeTraits<TBlocklike>::GetCacheItemHelper(block_type), create_cb,
        priority);
    if (cache_handle != nullptr) {
      block->SetCachedValue(
          reinterpret_cast<TBlocklike*>(block_cache->Value(cache_handle)),
          block_cache, cache_handle);
      return s;
    }
  }

  // If not found, search from the compressed block cache.
  assert(block->IsEmpty());

  if (block_cache_compressed == nullptr) {
    return s;
  }

  assert(!cache_key.empty());
  BlockContents contents;
  if (rep_->ioptions.lowest_used_cache_tier ==
      CacheTier::kNonVolatileBlockTier) {
    Cache::CreateCallback create_cb_special = GetCreateCallback<BlockContents>(
        read_amp_bytes_per_bit, statistics, using_zstd, filter_policy);
    block_cache_compressed_handle = block_cache_compressed->Lookup(
        cache_key,
        BlocklikeTraits<BlockContents>::GetCacheItemHelper(block_type),
        create_cb_special, priority, true);
  } else {
    block_cache_compressed_handle =
        block_cache_compressed->Lookup(cache_key, statistics);
  }

  // if we found in the compressed cache, then uncompress and insert into
  // uncompressed cache
  if (block_cache_compressed_handle == nullptr) {
    RecordTick(statistics, BLOCK_CACHE_COMPRESSED_MISS);
    return s;
  }

  // found compressed block
  RecordTick(statistics, BLOCK_CACHE_COMPRESSED_HIT);
  compressed_block = reinterpret_cast<BlockContents*>(
      block_cache_compressed->Value(block_cache_compressed_handle));
  CompressionType compression_type = GetBlockCompressionType(*compressed_block);
  assert(compression_type != kNoCompression);

  // Retrieve the uncompressed contents into a new buffer
  UncompressionContext context(compression_type);
  UncompressionInfo info(context, uncompression_dict, compression_type);
  s = UncompressBlockContents(
      info, compressed_block->data.data(), compressed_block->data.size(),
      &contents, rep_->table_options.format_version, rep_->ioptions,
      GetMemoryAllocator(rep_->table_options));

  // Insert uncompressed block into block cache, the priority is based on the
  // data block type.
  if (s.ok()) {
    std::unique_ptr<TBlocklike> block_holder(
        BlocklikeTraits<TBlocklike>::Create(
            std::move(contents), read_amp_bytes_per_bit, statistics,
            rep_->blocks_definitely_zstd_compressed,
            rep_->table_options.filter_policy.get()));  // uncompressed block

    if (block_cache != nullptr && block_holder->own_bytes() &&
        read_options.fill_cache) {
      size_t charge = block_holder->ApproximateMemoryUsage();
      Cache::Handle* cache_handle = nullptr;
      assert(false);
//      s = InsertEntryToCache(
//          rep_->ioptions.lowest_used_cache_tier, block_cache, cache_key,
//          BlocklikeTraits<TBlocklike>::GetCacheItemHelper(block_type),
//          block_holder, charge, &cache_handle, priority);
//      if (s.ok()) {
//        assert(cache_handle != nullptr);
//        block->SetCachedValue(block_holder.release(), block_cache,
//                              cache_handle);
//
//        UpdateCacheInsertionMetrics(block_type, get_context, charge,
//                                    s.IsOkOverwritten(), rep_->ioptions.stats);
//      } else {
//        RecordTick(statistics, BLOCK_CACHE_ADD_FAILURES);
//      }
    } else {
      block->SetOwnedValue(block_holder.release());
    }
  }

  // Release hold on compressed cache entry
  block_cache_compressed->Release(block_cache_compressed_handle);
  return s;

  return Status();
}

template <typename TBlocklike>
Status SeparatedBlockBasedTable::MaybeReadBlockAndLoadToCache(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, const UncompressionDict& uncompression_dict,
    const bool wait, const bool for_compaction,
    CachableEntry<TBlocklike>* block_entry, BlockType block_type,
    GetContext* get_context, BlockCacheLookupContext* lookup_context,
    BlockContents* contents) const {
  assert(block_entry != nullptr);
  const bool no_io = (ro.read_tier == kBlockCacheTier);
  Cache* block_cache = rep_->table_options.block_cache.get();
  Cache* block_cache_compressed =
      rep_->table_options.block_cache_compressed.get();

  // First, try to get the block from the cache
  //
  // If either block cache is enabled, we'll try to read from it.
  Status s;
  CacheKey key_data;
  Slice key;
  bool is_cache_hit = false;
  if (block_cache != nullptr || block_cache_compressed != nullptr) {
    // create key for block cache
    key_data = GetCacheKey(rep_->base_cache_key, handle);
    key = key_data.AsSlice();

    if (!contents) {
      s = GetDataBlockFromCache(key, block_cache, block_cache_compressed, ro,
                                block_entry, uncompression_dict, block_type,
                                wait, get_context);
      // Value could still be null at this point, so check the cache handle
      // and update the read pattern for prefetching
      if (block_entry->GetValue() || block_entry->GetCacheHandle()) {
        is_cache_hit = true;
        (void) is_cache_hit;
        if (prefetch_buffer) {
          // Update the block details so that PrefetchBuffer can use the read
          // pattern to determine if reads are sequential or not for
          // prefetching. It should also take in account blocks read from cache.
          prefetch_buffer->UpdateReadPattern(handle.offset(),
                                             BlockSizeWithTrailer(handle),
                                             ro.adaptive_readahead);
        }
      }
    }

    // Can't find the block from the cache. If I/O is allowed, read from the
    // file.
    if (block_entry->GetValue() == nullptr &&
        block_entry->GetCacheHandle() == nullptr && !no_io && ro.fill_cache) {
      Statistics* statistics = rep_->ioptions.stats;
      const bool maybe_compressed =
          block_type != BlockType::kFilter &&
          block_type != BlockType::kCompressionDictionary &&
          rep_->blocks_maybe_compressed;
      const bool do_uncompress = maybe_compressed && !block_cache_compressed;
      CompressionType raw_block_comp_type;
      BlockContents raw_block_contents;
      if (!contents) {
        Histograms histogram = for_compaction ? READ_BLOCK_COMPACTION_MICROS
                                              : READ_BLOCK_GET_MICROS;
        StopWatch sw(rep_->ioptions.clock, statistics, histogram);
        BlockFetcher block_fetcher(
            rep_->file.get(), prefetch_buffer, rep_->footer, ro, handle,
            &raw_block_contents, rep_->ioptions, do_uncompress,
            maybe_compressed, block_type, uncompression_dict,
            rep_->persistent_cache_options,
            GetMemoryAllocator(rep_->table_options),
            GetMemoryAllocatorForCompressedBlock(rep_->table_options));
        s = block_fetcher.ReadBlockContents();
        raw_block_comp_type = block_fetcher.get_compression_type();
        contents = &raw_block_contents;
        if (get_context) {
          switch (block_type) {
            case BlockType::kIndex:
              ++get_context->get_context_stats_.num_index_read;
              break;
            case BlockType::kFilter:
              ++get_context->get_context_stats_.num_filter_read;
              break;
            case BlockType::kData:
              ++get_context->get_context_stats_.num_data_read;
              break;
            default:
              break;
          }
        }
      } else {
        raw_block_comp_type = GetBlockCompressionType(*contents);
      }
      if (s.ok()) {
        // If filling cache is allowed and a cache is configured, try to put the
        // block to the cache.
        (void)raw_block_comp_type;
        //        s = PutDataBlockToCache(
        //            key, block_cache, block_cache_compressed, block_entry,
        //            contents, raw_block_comp_type, uncompression_dict,
        //            GetMemoryAllocator(rep_->table_options), block_type,
        //            get_context);
      }
    }
  }

  // Fill lookup_context.
  if (block_cache_tracer_ && block_cache_tracer_->is_tracing_enabled() &&
      lookup_context) {
  }

  assert(s.ok() || block_entry->GetValue() == nullptr);
  return s;
}

template <typename TBlocklike>
Status SeparatedBlockBasedTable::RetrieveBlock(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, const UncompressionDict& uncompression_dict,
    CachableEntry<TBlocklike>* block_entry, BlockType block_type,
    GetContext* get_context, BlockCacheLookupContext* lookup_context,
    bool for_compaction, bool use_cache, bool wait_for_cache) const {
  assert(block_entry);
  assert(block_entry->IsEmpty());

  Status s;
  if (use_cache) {
    s = MaybeReadBlockAndLoadToCache(
        prefetch_buffer, ro, handle, uncompression_dict, wait_for_cache,
        for_compaction, block_entry, block_type, get_context, lookup_context,
        /*contents=*/nullptr);

    if (!s.ok()) {
      return s;
    }

    if (block_entry->GetValue() != nullptr ||
        block_entry->GetCacheHandle() != nullptr) {
      assert(s.ok());
      return s;
    }
  }

  assert(block_entry->IsEmpty());

  const bool no_io = ro.read_tier == kBlockCacheTier;
  if (no_io) {
    return Status::Incomplete("no blocking io");
  }

  const bool maybe_compressed =
      block_type != BlockType::kFilter &&
      block_type != BlockType::kCompressionDictionary &&
      rep_->blocks_maybe_compressed;
  const bool do_uncompress = maybe_compressed;
  std::unique_ptr<TBlocklike> block;

  {
    Histograms histogram =
        for_compaction ? READ_BLOCK_COMPACTION_MICROS : READ_BLOCK_GET_MICROS;
    StopWatch sw(rep_->ioptions.clock, rep_->ioptions.stats, histogram);
    s = ReadBlockFromFile(
        rep_->file.get(), prefetch_buffer, rep_->footer, ro, handle, &block,
        rep_->ioptions, do_uncompress, maybe_compressed, block_type,
        uncompression_dict, rep_->persistent_cache_options,
        block_type == BlockType::kData
            ? rep_->table_options.read_amp_bytes_per_bit
            : 0,
        GetMemoryAllocator(rep_->table_options), for_compaction,
        rep_->blocks_definitely_zstd_compressed,
        rep_->table_options.filter_policy.get());

    if (get_context) {
      switch (block_type) {
        case BlockType::kIndex:
          ++(get_context->get_context_stats_.num_index_read);
          break;
        case BlockType::kFilter:
          ++(get_context->get_context_stats_.num_filter_read);
          break;
        case BlockType::kData:
          ++(get_context->get_context_stats_.num_data_read);
          break;
        default:
          break;
      }
    }
  }

  if (!s.ok()) {
    return s;
  }

  block_entry->SetOwnedValue(block.release());

  assert(s.ok());
  return s;
}
// Explicitly instantiate templates for both "blocklike" types we use.
// This makes it possible to keep the template definitions in the .cc file.
template Status SeparatedBlockBasedTable::RetrieveBlock<BlockContents>(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, const UncompressionDict& uncompression_dict,
    CachableEntry<BlockContents>* block_entry, BlockType block_type,
    GetContext* get_context, BlockCacheLookupContext* lookup_context,
    bool for_compaction, bool use_cache, bool wait_for_cache) const;

template Status SeparatedBlockBasedTable::RetrieveBlock<ParsedFullFilterBlock>(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, const UncompressionDict& uncompression_dict,
    CachableEntry<ParsedFullFilterBlock>* block_entry, BlockType block_type,
    GetContext* get_context, BlockCacheLookupContext* lookup_context,
    bool for_compaction, bool use_cache, bool wait_for_cache) const;

template Status SeparatedBlockBasedTable::RetrieveBlock<Block>(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, const UncompressionDict& uncompression_dict,
    CachableEntry<Block>* block_entry, BlockType block_type,
    GetContext* get_context, BlockCacheLookupContext* lookup_context,
    bool for_compaction, bool use_cache, bool wait_for_cache) const;

template Status SeparatedBlockBasedTable::RetrieveBlock<UncompressionDict>(
    FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
    const BlockHandle& handle, const UncompressionDict& uncompression_dict,
    CachableEntry<UncompressionDict>* block_entry, BlockType block_type,
    GetContext* get_context, BlockCacheLookupContext* lookup_context,
    bool for_compaction, bool use_cache, bool wait_for_cache) const;

}