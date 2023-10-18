#pragma once

#pragma once

#include <cstdint>

#include "cache/cache_key.h"
#include "db/range_tombstone_fragmenter.h"
#include "file/filename.h"
#include "rocksdb/slice_transform.h"
#include "rocksdb/table_properties.h"
#include "table/block_based/block.h"
#include "table/block_based/block_based_table_factory.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/block_type.h"
#include "table/block_based/cachable_entry.h"
#include "table/block_based/filter_block.h"
#include "table/block_based/uncompression_dict_reader.h"
#include "table/format.h"
#include "table/persistent_cache_options.h"
#include "table/table_properties_internal.h"
#include "table/table_reader.h"
#include "table/two_level_iterator.h"
#include "trace_replay/block_cache_tracer.h"

namespace ROCKSDB_NAMESPACE {

class TableReader;
class Footer;

class SeparatedBlockBasedTable : public TableReader {
 public:
  // 1-byte compression type + 32-bit checksum
  static constexpr size_t kBlockTrailerSize = 5;

  static Status Open(
      const ReadOptions& ro, const ImmutableOptions& ioptions,
      const EnvOptions& env_options,
      const BlockBasedTableOptions& table_options,
      const InternalKeyComparator& internal_key_comparator,
      std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
      std::unique_ptr<TableReader>* table_reader,
      const std::shared_ptr<const SliceTransform>& prefix_extractor = nullptr,
      bool prefetch_index_and_filter_in_cache = true, bool skip_filters = false,
      int level = -1, const bool immortal_table = false,
      const SequenceNumber largest_seqno = 0,
      bool force_direct_prefetch = false,
      TailPrefetchStats* tail_prefetch_stats = nullptr,
      BlockCacheTracer* const block_cache_tracer = nullptr,
      size_t max_file_size_for_l0_meta_pin = 0,
      const std::string& cur_db_session_id = "", uint64_t cur_file_num = 0);

  InternalIterator* NewIterator(const ReadOptions&,
                                const SliceTransform* prefix_extractor,
                                Arena* arena, bool skip_filters,
                                TableReaderCaller caller,
                                size_t compaction_readahead_size = 0,
                                bool allow_unprepared_value = false) override;

  Status Get(const ReadOptions& readOptions, const Slice& key,
             GetContext* get_context, const SliceTransform* prefix_extractor,
             bool skip_filters = false) override;

  uint64_t ApproximateOffsetOf(const Slice& key,
                               TableReaderCaller caller) override;

  uint64_t ApproximateSize(const Slice& start, const Slice& end,
                           TableReaderCaller caller) override;

  void SetupForCompaction() override;

  ~SeparatedBlockBasedTable();

  std::shared_ptr<const TableProperties> GetTableProperties() const override;

  size_t ApproximateMemoryUsage() const override;


  InternalIteratorBase<IndexValue>* NewIndexIterator(
      const ReadOptions& read_options, bool need_upper_bound_check,
      IndexBlockIter* input_iter, GetContext* get_context,
      BlockCacheLookupContext* lookup_context) const;

  // Read block cache from block caches (if set): block_cache and
  // block_cache_compressed.
  // On success, Status::OK with be returned and @block will be populated with
  // pointer to the block as well as its block handle.
  // @param uncompression_dict Data for presetting the compression library's
  //    dictionary.
  template <typename TBlocklike>
  Status GetDataBlockFromCache(const Slice& cache_key, Cache* block_cache,
                               Cache* block_cache_compressed,
                               const ReadOptions& read_options,
                               CachableEntry<TBlocklike>* block,
                               const UncompressionDict& uncompression_dict,
                               BlockType block_type, const bool wait,
                               GetContext* get_context) const;


  static void SetupBaseCacheKey(const TableProperties* properties,
                                const std::string& cur_db_session_id,
                                uint64_t cur_file_number, uint64_t file_size,
                                OffsetableCacheKey* out_base_cache_key,
                                bool* out_is_stable = nullptr);

  static CacheKey GetCacheKey(const OffsetableCacheKey& base_cache_key,
                              const BlockHandle& handle);

  static void UpdateCacheInsertionMetrics(BlockType block_type,
                                          GetContext* get_context, size_t usage,
                                          bool redundant,
                                          Statistics* const statistics);

  // Get the size to read from storage for a BlockHandle. size_t because we
  // are about to load into memory.
  static inline size_t BlockSizeWithTrailer(const BlockHandle& handle) {
    return static_cast<size_t>(handle.size() + kBlockTrailerSize);
  }

  // It's the caller's responsibility to make sure that this is
  // for raw block contents, which contains the compression
  // byte in the end.
  static inline CompressionType GetBlockCompressionType(const char* block_data,
                                                        size_t block_size) {
    return static_cast<CompressionType>(block_data[block_size]);
  }
  static inline CompressionType GetBlockCompressionType(
      const BlockContents& contents) {
    assert(contents.is_raw_block);
    return GetBlockCompressionType(contents.data.data(), contents.data.size());
  }

  struct Rep;

  Rep* get_rep() { return rep_; }
  const Rep* get_rep() const { return rep_; }

  // Similar to the above, with one crucial difference: it will retrieve the
  // block from the file even if there are no caches configured (assuming the
  // read options allow I/O).
  template <typename TBlocklike>
  Status RetrieveBlock(FilePrefetchBuffer* prefetch_buffer,
                       const ReadOptions& ro, const BlockHandle& handle,
                       const UncompressionDict& uncompression_dict,
                       CachableEntry<TBlocklike>* block_entry,
                       BlockType block_type, GetContext* get_context,
                       BlockCacheLookupContext* lookup_context,
                       bool for_compaction, bool use_cache,
                       bool wait_for_cache) const;

  template <typename TBlocklike>
  Status MaybeReadBlockAndLoadToCache(
      FilePrefetchBuffer* prefetch_buffer, const ReadOptions& ro,
      const BlockHandle& handle, const UncompressionDict& uncompression_dict,
      const bool wait, const bool for_compaction,
      CachableEntry<TBlocklike>* block_entry, BlockType block_type,
      GetContext* get_context, BlockCacheLookupContext* lookup_context,
      BlockContents* contents) const;

 protected:
  Rep* rep_;
  explicit SeparatedBlockBasedTable(Rep* rep, BlockCacheTracer* const block_cache_tracer)
      : rep_(rep), block_cache_tracer_(block_cache_tracer) {}
  // No copying allowed
  explicit SeparatedBlockBasedTable(const TableReader&) = delete;
  void operator=(const TableReader&) = delete;

 private:
  BlockCacheTracer* const block_cache_tracer_;

  friend class TableCache;

  // Create a index reader based on the index type stored in the table.
  // Optionally, user can pass a preloaded meta_index_iter for the index that
  // need to access extra meta blocks for index construction. This parameter
  // helps avoid re-reading meta index block if caller already created one.
  Status CreateIndexReader(const ReadOptions& ro,
                           FilePrefetchBuffer* prefetch_buffer,
                           InternalIterator* preloaded_meta_index_iter,
                           bool use_cache, bool prefetch, bool pin,
                           BlockCacheLookupContext* lookup_context,
                           std::unique_ptr<BlockBasedTable::IndexReader>* index_reader, std::unique_ptr<BlockBasedTable::IndexReader>* old_index_reader);

  // If force_direct_prefetch is true, always prefetching to RocksDB
  //    buffer, rather than calling RandomAccessFile::Prefetch().
  static Status PrefetchTail(
      const ReadOptions& ro, RandomAccessFileReader* file, uint64_t file_size,
      bool force_direct_prefetch, TailPrefetchStats* tail_prefetch_stats,
      const bool prefetch_all, const bool preload_all,
      std::unique_ptr<FilePrefetchBuffer>* prefetch_buffer);
  Status ReadMetaIndexBlock(const ReadOptions& ro,
                            FilePrefetchBuffer* prefetch_buffer,
                            std::unique_ptr<Block>* metaindex_block,
                            std::unique_ptr<InternalIterator>* iter);
  Status ReadPropertiesBlock(const ReadOptions& ro,
                             FilePrefetchBuffer* prefetch_buffer,
                             InternalIterator* meta_iter,
                             const SequenceNumber largest_seqno);
  Status ReadRangeDelBlock(const ReadOptions& ro,
                           FilePrefetchBuffer* prefetch_buffer,
                           InternalIterator* meta_iter,
                           const InternalKeyComparator& internal_comparator,
                           BlockCacheLookupContext* lookup_context);
  Status PrefetchIndexAndFilterBlocks(
      const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
      InternalIterator* meta_iter, SeparatedBlockBasedTable* new_table,
      bool prefetch_all, const BlockBasedTableOptions& table_options,
      const int level, size_t file_size, size_t max_file_size_for_l0_meta_pin,
      BlockCacheLookupContext* lookup_context);
};

// Stores all the properties associated with a BlockBasedTable.
// These are immutable.
struct SeparatedBlockBasedTable::Rep {
  Rep(const ImmutableOptions& _ioptions, const EnvOptions& _env_options,
      const BlockBasedTableOptions& _table_opt,
      const InternalKeyComparator& _internal_comparator, bool skip_filters,
      uint64_t _file_size, int _level, const bool _immortal_table)
      : ioptions(_ioptions),
        env_options(_env_options),
        table_options(_table_opt),
        filter_policy(skip_filters ? nullptr : _table_opt.filter_policy.get()),
        internal_comparator(_internal_comparator),
        filter_type(FilterType::kNoFilter),
        index_type(BlockBasedTableOptions::IndexType::kBinarySearch),
        hash_index_allow_collision(false),
        whole_key_filtering(_table_opt.whole_key_filtering),
        prefix_filtering(true),
        global_seqno(kDisableGlobalSequenceNumber),
        file_size(_file_size),
        level(_level),
        immortal_table(_immortal_table) {}
  ~Rep() { status.PermitUncheckedError(); }
  const ImmutableOptions& ioptions;
  const EnvOptions& env_options;
  const BlockBasedTableOptions table_options;
  const FilterPolicy* const filter_policy;
  const InternalKeyComparator& internal_comparator;
  Status status;
  std::unique_ptr<RandomAccessFileReader> file;
  OffsetableCacheKey base_cache_key;
  PersistentCacheOptions persistent_cache_options;

  // Footer contains the fixed table information
  Footer footer;

  std::unique_ptr<BlockBasedTable::IndexReader> index_reader;
  std::unique_ptr<FilterBlockReader> filter;
  std::unique_ptr<UncompressionDictReader> uncompression_dict_reader;

  enum class FilterType {
    kNoFilter,
    kFullFilter,
    kBlockFilter,
    kPartitionedFilter,
  };
  FilterType filter_type;
  BlockHandle filter_handle;
  BlockHandle compression_dict_handle;

  std::shared_ptr<const TableProperties> table_properties;
  BlockBasedTableOptions::IndexType index_type;
  bool hash_index_allow_collision;
  bool whole_key_filtering;
  bool prefix_filtering;

  std::unique_ptr<SliceTransform> internal_prefix_transform;
  std::shared_ptr<const SliceTransform> table_prefix_extractor;

  std::shared_ptr<const FragmentedRangeTombstoneList> fragmented_range_dels;

  // If global_seqno is used, all Keys in this file will have the same
  // seqno with value `global_seqno`.
  //
  // A value of kDisableGlobalSequenceNumber means that this feature is disabled
  // and every key have it's own seqno.
  SequenceNumber global_seqno;

  // Size of the table file on disk
  uint64_t file_size;

  // the level when the table is opened, could potentially change when trivial
  // move is involved
  int level;

  // If false, blocks in this file are definitely all uncompressed. Knowing this
  // before reading individual blocks enables certain optimizations.
  bool blocks_maybe_compressed = true;

  // If true, data blocks in this file are definitely ZSTD compressed. If false
  // they might not be. When false we skip creating a ZSTD digested
  // uncompression dictionary. Even if we get a false negative, things should
  // still work, just not as quickly.
  bool blocks_definitely_zstd_compressed = false;

  const bool immortal_table;
};

}  // namespace ROCKSDB_NAMESPACE