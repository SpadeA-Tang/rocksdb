#include "table/separated_block_based/separated_block_based_table_builder.h"

#include <assert.h>
#include <stdio.h>

#include <atomic>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>

#include "cache/cache_key.h"
#include "logging/logging.h"
#include "rocksdb/cache.h"
#include "rocksdb/comparator.h"
#include "rocksdb/env.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/flush_block_policy.h"
#include "rocksdb/merge_operator.h"
#include "rocksdb/table.h"
#include "rocksdb/types.h"
#include "table/block_based/block.h"
#include "table/block_based/block_based_filter_block.h"
#include "table/block_based/block_based_table_builder.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/block_like_traits.h"
#include "table/block_based/filter_block.h"
#include "table/block_based/filter_policy_internal.h"
#include "table/block_based/index_builder.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "table/block_based/partitioned_filter_block.h"
#include "table/format.h"
#include "table/meta_blocks.h"
#include "table/separated_block_based/separated_block_based_table_reader.h"
#include "table/table_builder.h"
#include "util/coding.h"
#include "util/compression.h"
#include "util/stop_watch.h"
#include "util/string_util.h"
#include "util/work_queue.h"

namespace ROCKSDB_NAMESPACE {

namespace {
constexpr size_t kBlockTrailerSize =
    SeparatedBlockBasedTable::kBlockTrailerSize;

// Create a filter block builder based on its type.
FilterBlockBuilder* CreateFilterBlockBuilder(
    const ImmutableCFOptions& /*opt*/, const MutableCFOptions& mopt,
    const FilterBuildingContext& context,
    const bool use_delta_encoding_for_index_values,
    PartitionedIndexBuilder* const p_index_builder) {
  const BlockBasedTableOptions& table_opt = context.table_options;
  assert(table_opt.filter_policy);  // precondition

  FilterBitsBuilder* filter_bits_builder =
      BloomFilterPolicy::GetBuilderFromContext(context);
  if (filter_bits_builder == nullptr) {
    return new BlockBasedFilterBlockBuilder(mopt.prefix_extractor.get(),
                                            table_opt);
  } else {
    if (table_opt.partition_filters) {
      assert(p_index_builder != nullptr);
      // Since after partition cut request from filter builder it takes time
      // until index builder actully cuts the partition, until the end of a
      // data block potentially with many keys, we take the lower bound as
      // partition size.
      assert(table_opt.block_size_deviation <= 100);
      auto partition_size =
          static_cast<uint32_t>(((table_opt.metadata_block_size *
                                  (100 - table_opt.block_size_deviation)) +
                                 99) /
                                100);
      partition_size = std::max(partition_size, static_cast<uint32_t>(1));
      return new PartitionedFilterBlockBuilder(
          mopt.prefix_extractor.get(), table_opt.whole_key_filtering,
          filter_bits_builder, table_opt.index_block_restart_interval,
          use_delta_encoding_for_index_values, p_index_builder, partition_size);
    } else {
      return new FullFilterBlockBuilder(mopt.prefix_extractor.get(),
                                        table_opt.whole_key_filtering,
                                        filter_bits_builder);
    }
  }
}
}

struct SeparatedBlockBasedTableBuilder::Rep {
  Rep(const BlockBasedTableOptions& table_opt, const TableBuilderOptions& tbo,
      const Comparator* user_comparator_, WritableFileWriter* f)
      : ioptions(tbo.ioptions),
        moptions(tbo.moptions),
        table_options(table_opt),
        user_comparator(user_comparator_),
        internal_comparator(tbo.internal_comparator),
        file(f),
        offset(0),
        old_offset(0),
        alignment(table_options.block_align
                      ? std::min(static_cast<size_t>(table_options.block_size),
                                 kDefaultPageSize)
                      : 0),
        data_block(table_options.block_restart_interval,
                   table_options.use_delta_encoding,
                   false /* use_value_delta_encoding */,
                   tbo.internal_comparator.user_comparator()
                           ->CanKeysWithDifferentByteContentsBeEqual()
                       ? BlockBasedTableOptions::kDataBlockBinarySearch
                       : table_options.data_block_index_type,
                   table_options.data_block_hash_table_util_ratio),
        old_data_block(table_options.block_restart_interval,
                       table_options.use_delta_encoding,
                       false /* use_value_delta_encoding */,
                       tbo.internal_comparator.user_comparator()
                               ->CanKeysWithDifferentByteContentsBeEqual()
                           ? BlockBasedTableOptions::kDataBlockBinarySearch
                           : table_options.data_block_index_type,
                       table_options.data_block_hash_table_util_ratio),
        internal_prefix_transform(tbo.moptions.prefix_extractor.get()),
        compression_type(tbo.compression_type),
        sample_for_compression(tbo.moptions.sample_for_compression),
        compressible_input_data_bytes(0),
        uncompressible_input_data_bytes(0),
        sampled_input_data_bytes(0),
        sampled_output_slow_data_bytes(0),
        sampled_output_fast_data_bytes(0),
        compression_opts(tbo.compression_opts),
        compression_dict(),
        compression_ctxs(tbo.compression_opts.parallel_threads),
        verify_ctxs(tbo.compression_opts.parallel_threads),
        verify_dict(),
        state((tbo.compression_opts.max_dict_bytes > 0) ? State::kBuffered
                                                        : State::kUnbuffered),
        use_delta_encoding_for_index_values(table_opt.format_version >= 4 &&
                                            !table_opt.block_align),
        reason(tbo.reason),
        flush_block_policy(
            table_options.flush_block_policy_factory->NewFlushBlockPolicy(
                table_options, data_block)),
        flush_old_block_policy(
            table_options.flush_block_policy_factory->NewFlushBlockPolicy(
                table_options, old_data_block)),
        status_ok(true),
        io_status_ok(true) {
    if (table_options.index_type ==
        BlockBasedTableOptions::kTwoLevelIndexSearch) {
      p_index_builder_ = PartitionedIndexBuilder::CreateIndexBuilder(
          &internal_comparator, use_delta_encoding_for_index_values,
          table_options);
      index_builder.reset(p_index_builder_);
      old_index_builder.reset(p_index_builder_);
    } else {
      index_builder.reset(IndexBuilder::CreateIndexBuilder(
          table_options.index_type, &internal_comparator,
          &this->internal_prefix_transform, use_delta_encoding_for_index_values,
          table_options));

      old_index_builder.reset(IndexBuilder::CreateIndexBuilder(
          table_options.index_type, &internal_comparator,
          &this->internal_prefix_transform, use_delta_encoding_for_index_values,
          table_options));
    }

    if (ioptions.optimize_filters_for_hits && tbo.is_bottommost) {
      filter_builder.reset();
    } else if (tbo.skip_filters) {
      filter_builder.reset();
    } else if (tbo.skip_filters) {
      filter_builder.reset();
    } else if (!table_options.filter_policy) {
      filter_builder.reset();
    } else {
      FilterBuildingContext filter_context(table_options);

      filter_context.info_log = ioptions.logger;
      filter_context.column_family_name = tbo.column_family_name;
      filter_context.reason = reason;

      // Only populate other fields if known to be in LSM rather than
      // generating external SST file
      if (reason != TableFileCreationReason::kMisc) {
        filter_context.compaction_style = ioptions.compaction_style;
        filter_context.num_levels = ioptions.num_levels;
        filter_context.level_at_creation = tbo.level_at_creation;
        filter_context.is_bottommost = tbo.is_bottommost;
        assert(filter_context.level_at_creation < filter_context.num_levels);
      }

      filter_builder.reset(CreateFilterBlockBuilder(
          ioptions, moptions, filter_context,
          use_delta_encoding_for_index_values, p_index_builder_));
    }
  }

  bool IsParallelCompressionEnabled() const { return false; }

  Status GetStatus() {
    // We need to make modifications of status visible when status_ok is set
    // to false, and this is ensured by status_mutex, so no special memory
    // order for status_ok is required.
    if (status_ok.load(std::memory_order_relaxed)) {
      return Status::OK();
    } else {
      return CopyStatus();
    }
  }

  Status CopyStatus() {
    std::lock_guard<std::mutex> lock(status_mutex);
    return status;
  }

  IOStatus GetIOStatus() {
    // We need to make modifications of io_status visible when status_ok is set
    // to false, and this is ensured by io_status_mutex, so no special memory
    // order for io_status_ok is required.
    if (io_status_ok.load(std::memory_order_relaxed)) {
      return IOStatus::OK();
    } else {
      return CopyIOStatus();
    }
  }

  IOStatus CopyIOStatus() {
    std::lock_guard<std::mutex> lock(io_status_mutex);
    return io_status;
  }

  uint64_t get_offset() { return offset.load(std::memory_order_relaxed); }
  void set_offset(uint64_t o) { offset.store(o, std::memory_order_relaxed); }

  // Never erase an existing status that is not OK.
  void SetStatus(Status s) {
    if (!s.ok() && status_ok.load(std::memory_order_relaxed)) {
      // Locking is an overkill for non compression_opts.parallel_threads
      // case but since it's unlikely that s is not OK, we take this cost
      // to be simplicity.
      std::lock_guard<std::mutex> lock(status_mutex);
      status = s;
      status_ok.store(false, std::memory_order_relaxed);
    }
  }

  void SetIOStatus(IOStatus ios) {
    if (!ios.ok() && io_status_ok.load(std::memory_order_relaxed)) {
      // Locking is an overkill for non compression_opts.parallel_threads
      // case but since it's unlikely that s is not OK, we take this cost
      // to be simplicity.
      std::lock_guard<std::mutex> lock(io_status_mutex);
      io_status = ios;
      io_status_ok.store(false, std::memory_order_relaxed);
    }
  }

  const ImmutableOptions ioptions;
  const MutableCFOptions moptions;
  const BlockBasedTableOptions table_options;
  const Comparator* user_comparator;
  const InternalKeyComparator& internal_comparator;
  WritableFileWriter* file;
  std::atomic<uint64_t> offset;
  std::atomic<uint64_t> old_offset;
  size_t alignment;
  BlockBuilder data_block;
  BlockBuilder old_data_block;
  std::vector<std::string> old_data_buffers;
  std::vector<std::pair<std::string, std::string>> old_index_metas;
  std::vector<BlockHandle> old_block_handles;

  InternalKeySliceTransform internal_prefix_transform;
  std::unique_ptr<IndexBuilder> index_builder;
  std::unique_ptr<IndexBuilder> old_index_builder;
  PartitionedIndexBuilder* p_index_builder_ = nullptr;

  std::string last_key;
  std::string last_new_key;
  std::string last_old_key;
  bool old_block_flushed = false;
  const Slice* first_key_in_next_block = nullptr;
  CompressionType compression_type;
  uint64_t sample_for_compression;
  const Slice* first_key_in_next_old_block = nullptr;
  std::atomic<uint64_t> compressible_input_data_bytes;
  std::atomic<uint64_t> uncompressible_input_data_bytes;
  std::atomic<uint64_t> sampled_input_data_bytes;
  std::atomic<uint64_t> sampled_output_slow_data_bytes;
  std::atomic<uint64_t> sampled_output_fast_data_bytes;
  CompressionOptions compression_opts;
  std::unique_ptr<CompressionDict> compression_dict;
  std::vector<std::unique_ptr<CompressionContext>> compression_ctxs;
  std::vector<std::unique_ptr<UncompressionContext>> verify_ctxs;
  std::unique_ptr<UncompressionDict> verify_dict;

  TableProperties props;

  enum class State {
    kBuffered,
    kUnbuffered,
    kClosed,
  };
  State state;

  BlockHandle pending_handle;  // Handle to add to index block
  const bool use_delta_encoding_for_index_values;
  std::unique_ptr<FilterBlockBuilder> filter_builder;
  OffsetableCacheKey base_cache_key;
  const TableFileCreationReason reason;

  std::string compressed_output;
  std::unique_ptr<FlushBlockPolicy> flush_block_policy;
  std::unique_ptr<FlushBlockPolicy> flush_old_block_policy;

  std::vector<std::unique_ptr<IntTblPropCollector>> table_properties_collectors;

  std::unique_ptr<ParallelCompressionRep> pc_rep;

 private:
  // Synchronize status & io_status accesses across threads from main thread,
  // compression thread and write thread in parallel compression.
  std::mutex status_mutex;
  std::atomic<bool> status_ok;
  std::atomic<bool> io_status_ok;
  Status status;
  std::mutex io_status_mutex;
  IOStatus io_status;
};

SeparatedBlockBasedTableBuilder::SeparatedBlockBasedTableBuilder(
    const BlockBasedTableOptions& table_options, const TableBuilderOptions& tbo,
    const Comparator* user_comparator, WritableFileWriter* file) {
  BlockBasedTableOptions sanitized_table_options(table_options);
  if (sanitized_table_options.format_version == 0 &&
      sanitized_table_options.checksum != kCRC32c) {
    ROCKS_LOG_WARN(
        tbo.ioptions.logger,
        "Silently converting format_version to 1 because checksum is "
        "non-default");
    // silently convert format_version to 1 to keep consistent with current
    // behavior
    sanitized_table_options.format_version = 1;
  }

  rep_ = new Rep(sanitized_table_options, tbo, user_comparator, file);

  if (rep_->filter_builder != nullptr) {
    rep_->filter_builder->StartBlock(0);
  }

  // Extremely large files use atypical cache key encoding, and we don't
  // know ahead of time how big the file will be. But assuming it's less
  // than 4TB, we will correctly predict the cache keys.
  BlockBasedTable::SetupBaseCacheKey(
      &rep_->props, tbo.db_session_id, tbo.cur_file_num,
      BlockBasedTable::kMaxFileSizeStandardEncoding, &rep_->base_cache_key);
}

SeparatedBlockBasedTableBuilder::~SeparatedBlockBasedTableBuilder() {
  // Catch errors where caller forgot to call Finish()
  assert(rep_->state == Rep::State::kClosed);
  delete rep_;
}

struct SeparatedBlockBasedTableBuilder::ParallelCompressionRep {
  // Keys is a wrapper of vector of strings avoiding
  // releasing string memories during vector clear()
  // in order to save memory allocation overhead
  class Keys {
   public:
    Keys() : keys_(kKeysInitSize), size_(0) {}
    void PushBack(const Slice& key) {
      if (size_ == keys_.size()) {
        keys_.emplace_back(key.data(), key.size());
      } else {
        keys_[size_].assign(key.data(), key.size());
      }
      size_++;
    }
    void SwapAssign(std::vector<std::string>& keys) {
      size_ = keys.size();
      std::swap(keys_, keys);
    }
    void Clear() { size_ = 0; }
    size_t Size() { return size_; }
    std::string& Back() { return keys_[size_ - 1]; }
    std::string& operator[](size_t idx) {
      assert(idx < size_);
      return keys_[idx];
    }

   private:
    const size_t kKeysInitSize = 32;
    std::vector<std::string> keys_;
    size_t size_;
  };
  std::unique_ptr<Keys> curr_block_keys;

  class BlockRepSlot;

  // BlockRep instances are fetched from and recycled to
  // block_rep_pool during parallel compression.
  struct BlockRep {
    Slice contents;
    Slice compressed_contents;
    std::unique_ptr<std::string> data;
    std::unique_ptr<std::string> compressed_data;
    CompressionType compression_type;
    std::unique_ptr<std::string> first_key_in_next_block;
    std::unique_ptr<Keys> keys;
    std::unique_ptr<BlockRepSlot> slot;
    Status status;
  };
  // Use a vector of BlockRep as a buffer for a determined number
  // of BlockRep structures. All data referenced by pointers in
  // BlockRep will be freed when this vector is destructed.
  using BlockRepBuffer = std::vector<BlockRep>;
  BlockRepBuffer block_rep_buf;
  // Use a thread-safe queue for concurrent access from block
  // building thread and writer thread.
  using BlockRepPool = WorkQueue<BlockRep*>;
  BlockRepPool block_rep_pool;

  // Use BlockRepSlot to keep block order in write thread.
  // slot_ will pass references to BlockRep
  class BlockRepSlot {
   public:
    BlockRepSlot() : slot_(1) {}
    template <typename T>
    void Fill(T&& rep) {
      slot_.push(std::forward<T>(rep));
    };
    void Take(BlockRep*& rep) { slot_.pop(rep); }

   private:
    // slot_ will pass references to BlockRep in block_rep_buf,
    // and those references are always valid before the destruction of
    // block_rep_buf.
    WorkQueue<BlockRep*> slot_;
  };

  // Compression queue will pass references to BlockRep in block_rep_buf,
  // and those references are always valid before the destruction of
  // block_rep_buf.
  using CompressQueue = WorkQueue<BlockRep*>;
  CompressQueue compress_queue;
  std::vector<port::Thread> compress_thread_pool;

  // Write queue will pass references to BlockRep::slot in block_rep_buf,
  // and those references are always valid before the corresponding
  // BlockRep::slot is destructed, which is before the destruction of
  // block_rep_buf.
  using WriteQueue = WorkQueue<BlockRepSlot*>;
  WriteQueue write_queue;
  std::unique_ptr<port::Thread> write_thread;

  // Estimate output file size when parallel compression is enabled. This is
  // necessary because compression & flush are no longer synchronized,
  // and BlockBasedTableBuilder::FileSize() is no longer accurate.
  // memory_order_relaxed suffices because accurate statistics is not required.
  class FileSizeEstimator {
   public:
    explicit FileSizeEstimator()
        : raw_bytes_compressed(0),
          raw_bytes_curr_block(0),
          raw_bytes_curr_block_set(false),
          raw_bytes_inflight(0),
          blocks_inflight(0),
          curr_compression_ratio(0),
          estimated_file_size(0) {}

    // Estimate file size when a block is about to be emitted to
    // compression thread
    void EmitBlock(uint64_t raw_block_size, uint64_t curr_file_size) {
      uint64_t new_raw_bytes_inflight =
          raw_bytes_inflight.fetch_add(raw_block_size,
                                       std::memory_order_relaxed) +
          raw_block_size;

      uint64_t new_blocks_inflight =
          blocks_inflight.fetch_add(1, std::memory_order_relaxed) + 1;

      estimated_file_size.store(
          curr_file_size +
              static_cast<uint64_t>(
                  static_cast<double>(new_raw_bytes_inflight) *
                  curr_compression_ratio.load(std::memory_order_relaxed)) +
              new_blocks_inflight * kBlockTrailerSize,
          std::memory_order_relaxed);
    }

    // Estimate file size when a block is already reaped from
    // compression thread
    void ReapBlock(uint64_t compressed_block_size, uint64_t curr_file_size) {
      assert(raw_bytes_curr_block_set);

      uint64_t new_raw_bytes_compressed =
          raw_bytes_compressed + raw_bytes_curr_block;
      assert(new_raw_bytes_compressed > 0);

      curr_compression_ratio.store(
          (curr_compression_ratio.load(std::memory_order_relaxed) *
               raw_bytes_compressed +
           compressed_block_size) /
              static_cast<double>(new_raw_bytes_compressed),
          std::memory_order_relaxed);
      raw_bytes_compressed = new_raw_bytes_compressed;

      uint64_t new_raw_bytes_inflight =
          raw_bytes_inflight.fetch_sub(raw_bytes_curr_block,
                                       std::memory_order_relaxed) -
          raw_bytes_curr_block;

      uint64_t new_blocks_inflight =
          blocks_inflight.fetch_sub(1, std::memory_order_relaxed) - 1;

      estimated_file_size.store(
          curr_file_size +
              static_cast<uint64_t>(
                  static_cast<double>(new_raw_bytes_inflight) *
                  curr_compression_ratio.load(std::memory_order_relaxed)) +
              new_blocks_inflight * kBlockTrailerSize,
          std::memory_order_relaxed);

      raw_bytes_curr_block_set = false;
    }

    void SetEstimatedFileSize(uint64_t size) {
      estimated_file_size.store(size, std::memory_order_relaxed);
    }

    uint64_t GetEstimatedFileSize() {
      return estimated_file_size.load(std::memory_order_relaxed);
    }

    void SetCurrBlockRawSize(uint64_t size) {
      raw_bytes_curr_block = size;
      raw_bytes_curr_block_set = true;
    }

   private:
    // Raw bytes compressed so far.
    uint64_t raw_bytes_compressed;
    // Size of current block being appended.
    uint64_t raw_bytes_curr_block;
    // Whether raw_bytes_curr_block has been set for next
    // ReapBlock call.
    bool raw_bytes_curr_block_set;
    // Raw bytes under compression and not appended yet.
    std::atomic<uint64_t> raw_bytes_inflight;
    // Number of blocks under compression and not appended yet.
    std::atomic<uint64_t> blocks_inflight;
    // Current compression ratio, maintained by BGWorkWriteRawBlock.
    std::atomic<double> curr_compression_ratio;
    // Estimated SST file size.
    std::atomic<uint64_t> estimated_file_size;
  };
  FileSizeEstimator file_size_estimator;

  // Facilities used for waiting first block completion. Need to Wait for
  // the completion of first block compression and flush to get a non-zero
  // compression ratio.
  std::atomic<bool> first_block_processed;
  std::condition_variable first_block_cond;
  std::mutex first_block_mutex;

  explicit ParallelCompressionRep(uint32_t parallel_threads)
      : curr_block_keys(new Keys()),
        block_rep_buf(parallel_threads),
        block_rep_pool(parallel_threads),
        compress_queue(parallel_threads),
        write_queue(parallel_threads),
        first_block_processed(false) {
    for (uint32_t i = 0; i < parallel_threads; i++) {
      block_rep_buf[i].contents = Slice();
      block_rep_buf[i].compressed_contents = Slice();
      block_rep_buf[i].data.reset(new std::string());
      block_rep_buf[i].compressed_data.reset(new std::string());
      block_rep_buf[i].compression_type = CompressionType();
      block_rep_buf[i].first_key_in_next_block.reset(new std::string());
      block_rep_buf[i].keys.reset(new Keys());
      block_rep_buf[i].slot.reset(new BlockRepSlot());
      block_rep_buf[i].status = Status::OK();
      block_rep_pool.push(&block_rep_buf[i]);
    }
  }

  ~ParallelCompressionRep() { block_rep_pool.finish(); }

  // Make a block prepared to be emitted to compression thread
  // Used in non-buffered mode
  BlockRep* PrepareBlock(CompressionType compression_type,
                         const Slice* first_key_in_next_block,
                         BlockBuilder* data_block) {
    BlockRep* block_rep =
        PrepareBlockInternal(compression_type, first_key_in_next_block);
    assert(block_rep != nullptr);
    data_block->SwapAndReset(*(block_rep->data));
    block_rep->contents = *(block_rep->data);
    std::swap(block_rep->keys, curr_block_keys);
    curr_block_keys->Clear();
    return block_rep;
  }

  // Used in EnterUnbuffered
  BlockRep* PrepareBlock(CompressionType compression_type,
                         const Slice* first_key_in_next_block,
                         std::string* data_block,
                         std::vector<std::string>* keys) {
    BlockRep* block_rep =
        PrepareBlockInternal(compression_type, first_key_in_next_block);
    assert(block_rep != nullptr);
    std::swap(*(block_rep->data), *data_block);
    block_rep->contents = *(block_rep->data);
    block_rep->keys->SwapAssign(*keys);
    return block_rep;
  }

  // Emit a block to compression thread
  void EmitBlock(BlockRep* block_rep) {
    assert(block_rep != nullptr);
    assert(block_rep->status.ok());
    if (!write_queue.push(block_rep->slot.get())) {
      return;
    }
    if (!compress_queue.push(block_rep)) {
      return;
    }

    if (!first_block_processed.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lock(first_block_mutex);
      first_block_cond.wait(lock, [this] {
        return first_block_processed.load(std::memory_order_relaxed);
      });
    }
  }

  // Reap a block from compression thread
  void ReapBlock(BlockRep* block_rep) {
    assert(block_rep != nullptr);
    block_rep->compressed_data->clear();
    block_rep_pool.push(block_rep);

    if (!first_block_processed.load(std::memory_order_relaxed)) {
      std::lock_guard<std::mutex> lock(first_block_mutex);
      first_block_processed.store(true, std::memory_order_relaxed);
      first_block_cond.notify_one();
    }
  }

 private:
  BlockRep* PrepareBlockInternal(CompressionType compression_type,
                                 const Slice* first_key_in_next_block) {
    BlockRep* block_rep = nullptr;
    block_rep_pool.pop(block_rep);
    assert(block_rep != nullptr);

    assert(block_rep->data);

    block_rep->compression_type = compression_type;

    if (first_key_in_next_block == nullptr) {
      block_rep->first_key_in_next_block.reset(nullptr);
    } else {
      block_rep->first_key_in_next_block->assign(
          first_key_in_next_block->data(), first_key_in_next_block->size());
    }

    return block_rep;
  }
};

// Only consider Unbuffered Now
void SeparatedBlockBasedTableBuilder::Add(const Slice& key,
                                          const Slice& value) {
  Rep* r = rep_;
  assert(rep_->state != Rep::State::kClosed);
  if (!ok()) return;
  ValueType value_type = ExtractValueType(key);
  if (IsValueType(value_type)) {
#ifndef NDEBUG
    if (r->props.num_entries > r->props.num_range_deletions) {
      assert(r->internal_comparator.Compare(key, Slice(r->last_key)) > 0);
    }
#endif  // !NDEBUG

    // New user key
    if (r->last_key.empty() ||
        r->user_comparator->CompareWithoutTimestamp(key, r->last_key) != 0) {
      auto should_flush = r->flush_block_policy->Update(key, value);
      if (should_flush) {
        assert(!r->data_block.empty());
        r->first_key_in_next_block = &key;
        FlushNewDataBlock();
        // todo: kBuffered

        if (ok() && r->state == Rep::State::kUnbuffered) {
          r->index_builder->AddIndexEntry(&r->last_new_key, &key,
                                          r->pending_handle);
        }
      }

      should_flush = r->flush_old_block_policy->Update(key, value);
      if (should_flush) {
        assert(!r->old_data_block.empty());
        r->first_key_in_next_old_block = &key;
        FlushOldDataBlock();
      }

      if (r->state == Rep::State::kUnbuffered) {
        if (r->filter_builder != nullptr) {
          size_t ts_sz =
              r->internal_comparator.user_comparator()->timestamp_size();
          r->filter_builder->Add(ExtractUserKeyAndStripTimestamp(key, ts_sz));
        }
      }

      r->data_block.AddWithLastKey(key, value, r->last_new_key);
      r->last_new_key.assign(key.data(), key.size());
    } else {
      if (r->old_block_flushed) {
        r->old_block_flushed = false;
        r->old_index_metas.push_back(std::make_pair(r->last_old_key, key.ToString()));
      }
      r->old_data_block.AddWithLastKey(key, value, r->last_old_key);
      r->last_old_key.assign(key.data(), key.size());
    }
    r->last_key.assign(key.data(), key.size());

    // todo: Table Collector and Data Compression
  } else if (value_type == kTypeRangeDeletion) {
  } else {
    assert(false);
  }

  r->props.num_entries++;
  r->props.raw_key_size += key.size();
  r->props.raw_value_size += value.size();
  if (value_type == kTypeDeletion || value_type == kTypeSingleDeletion) {
    r->props.num_deletions++;
  } else if (value_type == kTypeRangeDeletion) {
    r->props.num_deletions++;
    r->props.num_range_deletions++;
  } else if (value_type == kTypeMerge) {
    r->props.num_merge_operands++;
  }
}

void SeparatedBlockBasedTableBuilder::FlushNewDataBlock() {
  Rep* r = rep_;
  assert(rep_->state != Rep::State::kClosed);
  if (!ok()) return;
  if (r->data_block.empty()) return;
  WriteBlock(&r->data_block, &r->pending_handle, BlockType::kData);
}

void SeparatedBlockBasedTableBuilder::FlushOldDataBlock() {
  Rep* r = rep_;
  assert(rep_->state != Rep::State::kClosed);
  if (!ok()) return;
  if (r->old_data_block.empty()) return;

  r->old_data_block.Finish();
  std::string raw_block_contents;
  raw_block_contents.reserve(rep_->table_options.block_size);
  r->old_data_block.SwapAndReset(raw_block_contents);
  r->old_data_buffers.push_back(std::move(raw_block_contents));
  r->old_block_flushed = true;
}

void SeparatedBlockBasedTableBuilder::WriteBlock(BlockBuilder* block,
                                                 BlockHandle* handle,
                                                 BlockType block_type) {
  block->Finish();
  std::string raw_block_contents;
  raw_block_contents.reserve(rep_->table_options.block_size);
  block->SwapAndReset(raw_block_contents);
  // todo: kBuffered
  WriteBlock(raw_block_contents, handle, block_type);
}

void SeparatedBlockBasedTableBuilder::WriteBlock(
    const Slice& raw_block_contents, BlockHandle* handle,
    BlockType block_type) {
  Rep* r = rep_;
  assert(r->state == Rep::State::kUnbuffered);
  Slice block_contents;
  CompressionType type;
  Status compress_status;
  bool is_data_block = block_type == BlockType::kData;
  CompressAndVerifyBlock(raw_block_contents, is_data_block,
                         *(r->compression_ctxs[0]), r->verify_ctxs[0].get(),
                         &(r->compressed_output), &(block_contents), &type,
                         &compress_status);
  r->SetStatus(compress_status);
  if (!ok()) {
    return;
  }

  WriteRawBlock(block_contents, type, handle, block_type, &raw_block_contents);

  r->compressed_output.clear();
  if (is_data_block) {
    if (r->filter_builder != nullptr) {
      // todo: consider block-based-bloom-filter
      r->filter_builder->StartBlock(r->get_offset());
    }
    r->props.data_size = r->get_offset() + r->get_offset();
    ++r->props.num_data_blocks;
  }
}

Status SeparatedBlockBasedTableBuilder::InsertBlockInCacheHelper(
    const Slice& block_contents, const BlockHandle* handle,
    BlockType block_type, bool is_top_level_filter_block) {
  Status s;
  if (block_type == BlockType::kData || block_type == BlockType::kIndex) {
    s = InsertBlockInCache<Block>(block_contents, handle, block_type);
  } else if (block_type == BlockType::kFilter) {
    if (rep_->filter_builder->IsBlockBased()) {
      // for block-based filter which is deprecated.
      s = InsertBlockInCache<BlockContents>(block_contents, handle, block_type);
    } else if (is_top_level_filter_block) {
      // for top level filter block in partitioned filter.
      s = InsertBlockInCache<Block>(block_contents, handle, block_type);
    } else {
      // for second level partitioned filters and full filters.
      s = InsertBlockInCache<ParsedFullFilterBlock>(block_contents, handle,
                                                    block_type);
    }
  } else if (block_type == BlockType::kCompressionDictionary) {
    s = InsertBlockInCache<UncompressionDict>(block_contents, handle,
                                              block_type);
  }
  return s;
}

template <typename TBlocklike>
Status SeparatedBlockBasedTableBuilder::InsertBlockInCache(
    const Slice& block_contents, const BlockHandle* handle,
    BlockType block_type) {
  // Uncompressed regular block cache
  Cache* block_cache = rep_->table_options.block_cache.get();
  Status s;
  if (block_cache != nullptr) {
    size_t size = block_contents.size();
    auto buf = AllocateBlock(size, block_cache->memory_allocator());
    memcpy(buf.get(), block_contents.data(), size);
    BlockContents results(std::move(buf), size);

    CacheKey key = BlockBasedTable::GetCacheKey(rep_->base_cache_key, *handle);

    const size_t read_amp_bytes_per_bit =
        rep_->table_options.read_amp_bytes_per_bit;

    // TODO akanksha:: Dedup below code by calling
    // BlockBasedTable::PutDataBlockToCache.
    std::unique_ptr<TBlocklike> block_holder(
        BlocklikeTraits<TBlocklike>::Create(
            std::move(results), read_amp_bytes_per_bit,
            rep_->ioptions.statistics.get(),
            false /*rep_->blocks_definitely_zstd_compressed*/,
            rep_->table_options.filter_policy.get()));

    assert(block_holder->own_bytes());
    size_t charge = block_holder->ApproximateMemoryUsage();
    s = block_cache->Insert(
        key.AsSlice(), block_holder.get(),
        BlocklikeTraits<TBlocklike>::GetCacheItemHelper(block_type), charge,
        nullptr, Cache::Priority::LOW);

    if (s.ok()) {
      // Release ownership of block_holder.
      block_holder.release();
      BlockBasedTable::UpdateCacheInsertionMetrics(
          block_type, nullptr /*get_context*/, charge, s.IsOkOverwritten(),
          rep_->ioptions.stats);
    } else {
      RecordTick(rep_->ioptions.stats, BLOCK_CACHE_ADD_FAILURES);
    }
  }
  return s;
}

namespace {
// Delete the entry resided in the cache.
template <class Entry>
void DeleteEntryCached(const Slice& /*key*/, void* value) {
  auto entry = reinterpret_cast<Entry*>(value);
  delete entry;
}
}  // namespace

Status SeparatedBlockBasedTableBuilder::InsertBlockInCompressedCache(
    const Slice& block_contents, const CompressionType type,
    const BlockHandle* handle) {
  Rep* r = rep_;
  Cache* block_cache_compressed = r->table_options.block_cache_compressed.get();
  Status s;
  if (type != kNoCompression && block_cache_compressed != nullptr) {
    size_t size = block_contents.size();

    auto ubuf =
        AllocateBlock(size + 1, block_cache_compressed->memory_allocator());
    memcpy(ubuf.get(), block_contents.data(), size);
    ubuf[size] = type;

    BlockContents* block_contents_to_cache =
        new BlockContents(std::move(ubuf), size);
#ifndef NDEBUG
    block_contents_to_cache->is_raw_block = true;
#endif  // NDEBUG

    CacheKey key = BlockBasedTable::GetCacheKey(rep_->base_cache_key, *handle);

    s = block_cache_compressed->Insert(
        key.AsSlice(), block_contents_to_cache,
        block_contents_to_cache->ApproximateMemoryUsage(),
        &DeleteEntryCached<BlockContents>);
    if (s.ok()) {
      RecordTick(rep_->ioptions.stats, BLOCK_CACHE_COMPRESSED_ADD);
    } else {
      RecordTick(rep_->ioptions.stats, BLOCK_CACHE_COMPRESSED_ADD_FAILURES);
    }
    // Invalidate OS cache.
    // todo: consider old_block_cache
    r->file->InvalidateCache(static_cast<size_t>(r->get_offset()), size)
        .PermitUncheckedError();
  }
  return s;
}

void SeparatedBlockBasedTableBuilder::WriteFilterBlock(
    MetaIndexBuilder* meta_index_builder) {
  BlockHandle filter_block_handle;
  bool empty_filter_block =
      (rep_->filter_builder == nullptr || rep_->filter_builder->IsEmpty());
  if (ok() && !empty_filter_block) {
    rep_->props.num_filter_entries +=
        rep_->filter_builder->EstimateEntriesAdded();
    Status s = Status::Incomplete();
    while (ok() && s.IsIncomplete()) {
      // filter_data is used to store the transferred filter data payload from
      // FilterBlockBuilder and deallocate the payload by going out of scope.
      // Otherwise, the payload will unnecessarily remain until
      // BlockBasedTableBuilder is deallocated.
      //
      // See FilterBlockBuilder::Finish() for more on the difference in
      // transferred filter data payload among different FilterBlockBuilder
      // subtypes.
      std::unique_ptr<const char[]> filter_data;
      Slice filter_content =
          rep_->filter_builder->Finish(filter_block_handle, &s, &filter_data);
      assert(s.ok() || s.IsIncomplete());
      rep_->props.filter_size += filter_content.size();

      // TODO: Refactor code so that BlockType can determine both the C++ type
      // of a block cache entry (TBlocklike) and the CacheEntryRole while
      // inserting blocks in cache.
      bool top_level_filter_block = false;
      if (s.ok() && rep_->table_options.partition_filters &&
          !rep_->filter_builder->IsBlockBased()) {
        top_level_filter_block = true;
      }
      WriteRawBlock(filter_content, kNoCompression, &filter_block_handle,
                    BlockType::kFilter, nullptr /*raw_contents*/,
                    top_level_filter_block);
    }
    rep_->filter_builder->ResetFilterBitsBuilder();
  }
  if (ok() && !empty_filter_block) {
    // Add mapping from "<filter_block_prefix>.Name" to location
    // of filter data.
    std::string key;
    if (rep_->filter_builder->IsBlockBased()) {
      key = BlockBasedTable::kFilterBlockPrefix;
    } else {
      key = rep_->table_options.partition_filters
                ? BlockBasedTable::kPartitionedFilterBlockPrefix
                : BlockBasedTable::kFullFilterBlockPrefix;
    }
    key.append(rep_->table_options.filter_policy->Name());
    meta_index_builder->Add(key, filter_block_handle);
  }
}

void SeparatedBlockBasedTableBuilder::WriteIndexBlock(
    MetaIndexBuilder* meta_index_builder, BlockHandle* index_block_handle,
    IndexBuilder* index_builder) {
  IndexBuilder::IndexBlocks index_blocks;
  auto index_builder_status = index_builder->Finish(&index_blocks);
  std::cout << "WriteIndexBlock, data "
            << index_blocks.index_block_contents.ToString(true) << std::endl;
  if (index_builder_status.IsIncomplete()) {
    // We we have more than one index partition then meta_blocks are not
    // supported for the index. Currently meta_blocks are used only by
    // HashIndexBuilder which is not multi-partition.
    assert(index_blocks.meta_blocks.empty());
  } else if (ok() && !index_builder_status.ok()) {
    rep_->SetStatus(index_builder_status);
  }
  if (ok()) {
    for (const auto& item : index_blocks.meta_blocks) {
      BlockHandle block_handle;
      WriteBlock(item.second, &block_handle, BlockType::kIndex);
      if (!ok()) {
        break;
      }
      meta_index_builder->Add(item.first, block_handle);
    }
  }
  if (ok()) {
    if (rep_->table_options.enable_index_compression) {
      WriteBlock(index_blocks.index_block_contents, index_block_handle,
                 BlockType::kIndex);
    } else {
      WriteRawBlock(index_blocks.index_block_contents, kNoCompression,
                    index_block_handle, BlockType::kIndex);
    }
  }
  // If there are more index partitions, finish them and write them out
  if (index_builder_status.IsIncomplete()) {
    bool index_building_finished = false;
    while (ok() && !index_building_finished) {
      Status s = index_builder->Finish(&index_blocks, *index_block_handle);
      if (s.ok()) {
        index_building_finished = true;
      } else if (s.IsIncomplete()) {
        // More partitioned index after this one
        assert(!index_building_finished);
      } else {
        // Error
        rep_->SetStatus(s);
        return;
      }

      if (rep_->table_options.enable_index_compression) {
        WriteBlock(index_blocks.index_block_contents, index_block_handle,
                   BlockType::kIndex);
      } else {
        WriteRawBlock(index_blocks.index_block_contents, kNoCompression,
                      index_block_handle, BlockType::kIndex);
      }
      // The last index_block_handle will be for the partition index block
    }
  }
}

void SeparatedBlockBasedTableBuilder::WritePropertiesBlock(
    MetaIndexBuilder* meta_index_builder) {
  BlockHandle properties_block_handle;
  if (ok()) {
    PropertyBlockBuilder property_block_builder;
    rep_->props.filter_policy_name =
        rep_->table_options.filter_policy != nullptr
            ? rep_->table_options.filter_policy->Name()
            : "";
    rep_->props.index_size =
        rep_->index_builder->IndexSize() + kBlockTrailerSize;
    rep_->props.comparator_name = rep_->ioptions.user_comparator != nullptr
                                      ? rep_->ioptions.user_comparator->Name()
                                      : "nullptr";
    rep_->props.merge_operator_name =
        rep_->ioptions.merge_operator != nullptr
            ? rep_->ioptions.merge_operator->Name()
            : "nullptr";
    rep_->props.compression_name =
        CompressionTypeToString(rep_->compression_type);
    rep_->props.compression_options =
        CompressionOptionsToString(rep_->compression_opts);
    rep_->props.prefix_extractor_name =
        rep_->moptions.prefix_extractor != nullptr
            ? rep_->moptions.prefix_extractor->AsString()
            : "nullptr";
    std::string property_collectors_names = "[";
    for (size_t i = 0;
         i < rep_->ioptions.table_properties_collector_factories.size(); ++i) {
      if (i != 0) {
        property_collectors_names += ",";
      }
      property_collectors_names +=
          rep_->ioptions.table_properties_collector_factories[i]->Name();
    }
    property_collectors_names += "]";
    rep_->props.property_collectors_names = property_collectors_names;
    if (rep_->table_options.index_type ==
        BlockBasedTableOptions::kTwoLevelIndexSearch) {
      assert(rep_->p_index_builder_ != nullptr);
      rep_->props.index_partitions = rep_->p_index_builder_->NumPartitions();
      rep_->props.top_level_index_size =
          rep_->p_index_builder_->TopLevelIndexSize(rep_->offset);
    }
    rep_->props.index_key_is_user_key =
        !rep_->index_builder->seperator_is_key_plus_seq();
    rep_->props.index_value_is_delta_encoded =
        rep_->use_delta_encoding_for_index_values;
    if (rep_->sampled_input_data_bytes > 0) {
      rep_->props.slow_compression_estimated_data_size = static_cast<uint64_t>(
          static_cast<double>(rep_->sampled_output_slow_data_bytes) /
              rep_->sampled_input_data_bytes *
              rep_->compressible_input_data_bytes +
          rep_->uncompressible_input_data_bytes + 0.5);
      rep_->props.fast_compression_estimated_data_size = static_cast<uint64_t>(
          static_cast<double>(rep_->sampled_output_fast_data_bytes) /
              rep_->sampled_input_data_bytes *
              rep_->compressible_input_data_bytes +
          rep_->uncompressible_input_data_bytes + 0.5);
    } else if (rep_->sample_for_compression > 0) {
      // We tried to sample but none were found. Assume worst-case (compression
      // ratio 1.0) so data is complete and aggregatable.
      rep_->props.slow_compression_estimated_data_size =
          rep_->compressible_input_data_bytes +
          rep_->uncompressible_input_data_bytes;
      rep_->props.fast_compression_estimated_data_size =
          rep_->compressible_input_data_bytes +
          rep_->uncompressible_input_data_bytes;
    }

    // Add basic properties
    property_block_builder.AddTableProperty(rep_->props);

    // Add use collected properties
    NotifyCollectTableCollectorsOnFinish(rep_->table_properties_collectors,
                                         rep_->ioptions.logger,
                                         &property_block_builder);

    Slice block_data = property_block_builder.Finish();
    TEST_SYNC_POINT_CALLBACK(
        "BlockBasedTableBuilder::WritePropertiesBlock:BlockData", &block_data);
    WriteRawBlock(block_data, kNoCompression, &properties_block_handle,
                  BlockType::kProperties);
  }
  if (ok()) {
#ifndef NDEBUG
    {
      uint64_t props_block_offset = properties_block_handle.offset();
      uint64_t props_block_size = properties_block_handle.size();
      TEST_SYNC_POINT_CALLBACK(
          "BlockBasedTableBuilder::WritePropertiesBlock:GetPropsBlockOffset",
          &props_block_offset);
      TEST_SYNC_POINT_CALLBACK(
          "BlockBasedTableBuilder::WritePropertiesBlock:GetPropsBlockSize",
          &props_block_size);
    }
#endif  // !NDEBUG

    const std::string* properties_block_meta = &kPropertiesBlockName;
    TEST_SYNC_POINT_CALLBACK(
        "BlockBasedTableBuilder::WritePropertiesBlock:Meta",
        &properties_block_meta);
    meta_index_builder->Add(*properties_block_meta, properties_block_handle);
  }
}

void SeparatedBlockBasedTableBuilder::WriteCompressionDictBlock(
    MetaIndexBuilder* meta_index_builder) {
  if (rep_->compression_dict != nullptr &&
      rep_->compression_dict->GetRawDict().size()) {
    BlockHandle compression_dict_block_handle;
    if (ok()) {
      WriteRawBlock(rep_->compression_dict->GetRawDict(), kNoCompression,
                    &compression_dict_block_handle,
                    BlockType::kCompressionDictionary);
#ifndef NDEBUG
      Slice compression_dict = rep_->compression_dict->GetRawDict();
      TEST_SYNC_POINT_CALLBACK(
          "BlockBasedTableBuilder::WriteCompressionDictBlock:RawDict",
          &compression_dict);
#endif  // NDEBUG
    }
    if (ok()) {
      meta_index_builder->Add(kCompressionDictBlockName,
                              compression_dict_block_handle);
    }
  }
}

void SeparatedBlockBasedTableBuilder::WriteRangeDelBlock(
    MetaIndexBuilder* meta_index_builder) {}

void SeparatedBlockBasedTableBuilder::WriteFooter(
    BlockHandle& metaindex_block_handle, BlockHandle& index_block_handle,
    BlockHandle& old_index_block_handle) {
  Rep* r = rep_;
  // this is guaranteed by BlockBasedTableBuilder's constructor
  assert(r->table_options.checksum == kCRC32c ||
         r->table_options.format_version != 0);
  assert(ok());

  FooterBuilder footer;
  footer.Build(kBlockBasedTableMagicNumber, r->table_options.format_version,
               r->get_offset(), r->table_options.checksum,
               metaindex_block_handle, index_block_handle,
               &old_index_block_handle);
  IOStatus ios = r->file->Append(footer.GetSlice());
  if (ios.ok()) {
    r->set_offset(r->get_offset() + footer.GetSlice().size());
  } else {
    r->SetIOStatus(ios);
    r->SetStatus(ios);
  }
}

Status SeparatedBlockBasedTableBuilder::Finish() {
  Rep* r = rep_;
  assert(r->state != Rep::State::kClosed);
  bool empty_data_block = r->data_block.empty();
  r->first_key_in_next_block = nullptr;
  FlushNewDataBlock();
  if (ok() && !empty_data_block) {
    r->index_builder->AddIndexEntry(
        &r->last_key, nullptr /* no next data block */, r->pending_handle);
  }

  if (r->old_block_flushed) {
    r->old_index_metas.emplace_back(r->last_old_key, std::string{});
  } else {
    r->first_key_in_next_old_block = nullptr;
    FlushOldDataBlock();
    if (r->old_block_flushed) {
      r->old_index_metas.emplace_back(r->last_old_key, std::string{});
    }
  }
  assert(rep_->old_data_buffers.size() == rep_->old_index_metas.size());
  BlockHandle h;
  for (size_t i = 0; i< rep_->old_data_buffers.size(); i++) {
    const std::string& block_contents = rep_->old_data_buffers[i];
    WriteBlock(block_contents, &h, BlockType::kData);
    auto& p = rep_->old_index_metas[i];
    std::string* last_key = &p.first;
    Slice next_key = Slice(p.second.data(), p.second.size());
    r->old_index_builder->AddIndexEntry(last_key, next_key.empty() ? nullptr: &next_key,
                                        h);
    r->pending_handle = h;
  }

  // Write meta blocks, metaindex block and footer in the following order.
  //    1. [meta block: filter]
  //    2. [meta block: index]
  //    3. [meta block: old_data_index]
  //    4. [meta block: compression dictionary]
  //    5. [meta block: range deletion tombstone]
  //    6. [meta block: properties]
  //    7. [metaindex block]
  //    8. Footer
  BlockHandle metaindex_block_handle, index_block_handle,
      old_index_block_handle;
  MetaIndexBuilder meta_index_builder;
  WriteFilterBlock(&meta_index_builder);
  WriteIndexBlock(&meta_index_builder, &index_block_handle,
                  rep_->index_builder.get());
  WriteIndexBlock(&meta_index_builder, &old_index_block_handle,
                  rep_->old_index_builder.get());
  WriteCompressionDictBlock(&meta_index_builder);
  WriteRangeDelBlock(&meta_index_builder);
  WritePropertiesBlock(&meta_index_builder);
  if (ok()) {
    // flush the meta index block
    WriteRawBlock(meta_index_builder.Finish(), kNoCompression,
                  &metaindex_block_handle, BlockType::kMetaIndex);
  }
  if (ok()) {
    WriteFooter(metaindex_block_handle, index_block_handle,
                old_index_block_handle);
  }
  assert(r->get_offset() == r->file->GetFileSize());
  r->state = Rep::State::kClosed;
  return Status{};
}

void SeparatedBlockBasedTableBuilder::Abandon() {
  assert(rep_->state != Rep::State::kClosed);
  rep_->state = Rep::State::kClosed;
}

uint64_t SeparatedBlockBasedTableBuilder::NumEntries() const {
  return rep_->props.num_entries;
}

bool SeparatedBlockBasedTableBuilder::IsEmpty() const {
  return rep_->props.num_entries == 0 && rep_->props.num_range_deletions == 0;
}

uint64_t SeparatedBlockBasedTableBuilder::FileSize() const {
  return rep_->offset;
}

TableProperties SeparatedBlockBasedTableBuilder::GetTableProperties() const {
  TableProperties ret = rep_->props;
  for (const auto& collector : rep_->table_properties_collectors) {
    for (const auto& prop : collector->GetReadableProperties()) {
      ret.readable_properties.insert(prop);
    }
    collector->Finish(&ret.user_collected_properties).PermitUncheckedError();
  }
  return ret;
}

std::string SeparatedBlockBasedTableBuilder::GetFileChecksum() const {
  if (rep_->file != nullptr) {
    return rep_->file->GetFileChecksum();
  } else {
    return kUnknownFileChecksum;
  }
}

const char* SeparatedBlockBasedTableBuilder::GetFileChecksumFuncName() const {
  if (rep_->file != nullptr) {
    return rep_->file->GetFileChecksumFuncName();
  } else {
    return kUnknownFileChecksumFuncName;
  }
}

Status SeparatedBlockBasedTableBuilder::status() const {
  return rep_->GetStatus();
}

IOStatus SeparatedBlockBasedTableBuilder::io_status() const {
  return rep_->GetIOStatus();
}

void SeparatedBlockBasedTableBuilder::WriteRawBlock(
    const Slice& block_contents, CompressionType type, BlockHandle* handle,
    BlockType block_type, const Slice* raw_block_contents,
    bool is_top_level_filter_block) {
  //  std::cout << block_contents.ToString(true) << std::endl;
  //  std::cout << "====================" << std::endl;
  //  std::cout << raw_block_contents->ToString(true) << std::endl;
  Rep* r = rep_;
  bool is_data_block = block_type == BlockType::kData;
  Status s = Status::OK();
  IOStatus io_s = IOStatus::OK();
  StopWatch sw(r->ioptions.clock, r->ioptions.stats, WRITE_RAW_BLOCK_MICROS);
  handle->set_offset(r->get_offset());
  handle->set_size(block_contents.size());
  assert(status().ok());
  assert(io_status().ok());
  io_s = r->file->Append(block_contents);
  if (io_s.ok()) {
    std::array<char, kBlockTrailerSize> trailer;
    trailer[0] = type;
    uint32_t checksum = ComputeBuiltinChecksumWithLastByte(
        r->table_options.checksum, block_contents.data(), block_contents.size(),
        /*last_byte*/ type);
    EncodeFixed32(trailer.data() + 1, checksum);

    assert(io_s.ok());
    TEST_SYNC_POINT_CALLBACK(
        "BlockBasedTableBuilder::WriteRawBlock:TamperWithChecksum",
        trailer.data());
      io_s = r->file->Append(Slice(trailer.data(), trailer.size()));
    if (io_s.ok()) {
      assert(s.ok());
      bool warm_cache;
      switch (r->table_options.prepopulate_block_cache) {
        case BlockBasedTableOptions::PrepopulateBlockCache::kFlushOnly:
          warm_cache = (r->reason == TableFileCreationReason::kFlush);
          break;
        case BlockBasedTableOptions::PrepopulateBlockCache::kDisable:
          warm_cache = false;
          break;
        default:
          // missing case
          assert(false);
          warm_cache = false;
      }
      if (warm_cache) {
        if (type == kNoCompression) {
          s = InsertBlockInCacheHelper(block_contents, handle, block_type,
                                       is_top_level_filter_block);
        } else if (raw_block_contents != nullptr) {
          s = InsertBlockInCacheHelper(*raw_block_contents, handle, block_type,
                                       is_top_level_filter_block);
        }
        if (!s.ok()) {
          r->SetStatus(s);
        }
      }
      // TODO:: Should InsertBlockInCompressedCache take into account error from
      // InsertBlockInCache or ignore and overwrite it.
      s = InsertBlockInCompressedCache(block_contents, type, handle);
      if (!s.ok()) {
        r->SetStatus(s);
      }
    } else {
      r->SetIOStatus(io_s);
    }
    if (s.ok() && io_s.ok()) {
      r->set_offset(r->get_offset() + block_contents.size() +
                    kBlockTrailerSize);
      if (r->table_options.block_align && is_data_block) {
        size_t pad_bytes =
            (r->alignment - ((block_contents.size() + kBlockTrailerSize) &
                             (r->alignment - 1))) &
            (r->alignment - 1);
        io_s = r->file->Pad(pad_bytes);
        if (io_s.ok()) {
          r->set_offset(r->get_offset() + pad_bytes);
        } else {
          r->SetIOStatus(io_s);
        }
      }
      if (r->IsParallelCompressionEnabled()) {
        if (is_data_block) {
          r->pc_rep->file_size_estimator.ReapBlock(block_contents.size(),
                                                   r->get_offset());
        } else {
          r->pc_rep->file_size_estimator.SetEstimatedFileSize(r->get_offset());
        }
      }
    }
  } else {
    r->SetIOStatus(io_s);
  }
  if (!io_s.ok() && s.ok()) {
    r->SetStatus(io_s);
  }
}

void SeparatedBlockBasedTableBuilder::CompressAndVerifyBlock(
    const Slice& raw_block_contents, bool is_data_block,
    const CompressionContext& compression_ctx, UncompressionContext* verify_ctx,
    std::string* compressed_output, Slice* block_contents,
    CompressionType* type, Status* out_status) {
  // File format contains a sequence of blocks where each block has:
  //    block_data: uint8[n]
  //    type: uint8
  //    crc: uint32
  Rep* r = rep_;
  bool is_status_ok = ok();
  if (!r->IsParallelCompressionEnabled()) {
    assert(is_status_ok);
  }

  *type = r->compression_type;
  uint64_t sample_for_compression = r->sample_for_compression;
  bool abort_compression = false;

  StopWatchNano timer(
      r->ioptions.clock,
      ShouldReportDetailedTime(r->ioptions.env, r->ioptions.stats));

  if (is_status_ok && raw_block_contents.size() < kCompressionSizeLimit) {
    if (is_data_block) {
      r->compressible_input_data_bytes.fetch_add(raw_block_contents.size(),
                                                 std::memory_order_relaxed);
    }
    const CompressionDict* compression_dict;
    if (!is_data_block || r->compression_dict == nullptr) {
      compression_dict = &CompressionDict::GetEmptyDict();
    } else {
      compression_dict = r->compression_dict.get();
    }
    assert(compression_dict != nullptr);
    CompressionInfo compression_info(r->compression_opts, compression_ctx,
                                     *compression_dict, *type,
                                     sample_for_compression);

    std::string sampled_output_fast;
    std::string sampled_output_slow;
    *block_contents = CompressBlock(
        raw_block_contents, compression_info, type,
        r->table_options.format_version, is_data_block /* do_sample */,
        compressed_output, &sampled_output_fast, &sampled_output_slow);

    if (sampled_output_slow.size() > 0 || sampled_output_fast.size() > 0) {
      // Currently compression sampling is only enabled for data block.
      assert(is_data_block);
      r->sampled_input_data_bytes.fetch_add(raw_block_contents.size(),
                                            std::memory_order_relaxed);
      r->sampled_output_slow_data_bytes.fetch_add(sampled_output_slow.size(),
                                                  std::memory_order_relaxed);
      r->sampled_output_fast_data_bytes.fetch_add(sampled_output_fast.size(),
                                                  std::memory_order_relaxed);
    }
    // notify collectors on block add
    NotifyCollectTableCollectorsOnBlockAdd(
        r->table_properties_collectors, raw_block_contents.size(),
        sampled_output_fast.size(), sampled_output_slow.size());

    // Some of the compression algorithms are known to be unreliable. If
    // the verify_compression flag is set then try to de-compress the
    // compressed data and compare to the input.
    if (*type != kNoCompression && r->table_options.verify_compression) {
      // Retrieve the uncompressed contents into a new buffer
      const UncompressionDict* verify_dict;
      if (!is_data_block || r->verify_dict == nullptr) {
        verify_dict = &UncompressionDict::GetEmptyDict();
      } else {
        verify_dict = r->verify_dict.get();
      }
      assert(verify_dict != nullptr);
      BlockContents contents;
      UncompressionInfo uncompression_info(*verify_ctx, *verify_dict,
                                           r->compression_type);
      Status stat = UncompressBlockContentsForCompressionType(
          uncompression_info, block_contents->data(), block_contents->size(),
          &contents, r->table_options.format_version, r->ioptions);

      if (stat.ok()) {
        bool compressed_ok = contents.data.compare(raw_block_contents) == 0;
        if (!compressed_ok) {
          // The result of the compression was invalid. abort.
          abort_compression = true;
          ROCKS_LOG_ERROR(r->ioptions.logger,
                          "Decompressed block did not match raw block");
          *out_status =
              Status::Corruption("Decompressed block did not match raw block");
        }
      } else {
        // Decompression reported an error. abort.
        *out_status = Status::Corruption(std::string("Could not decompress: ") +
                                         stat.getState());
        abort_compression = true;
      }
    }
  } else {
    // Block is too big to be compressed.
    if (is_data_block) {
      r->uncompressible_input_data_bytes.fetch_add(raw_block_contents.size(),
                                                   std::memory_order_relaxed);
    }
    abort_compression = true;
  }
  if (is_data_block) {
    r->uncompressible_input_data_bytes.fetch_add(kBlockTrailerSize,
                                                 std::memory_order_relaxed);
  }

  // Abort compression if the block is too big, or did not pass
  // verification.
  if (abort_compression) {
    RecordTick(r->ioptions.stats, NUMBER_BLOCK_NOT_COMPRESSED);
    *type = kNoCompression;
    *block_contents = raw_block_contents;
  } else if (*type != kNoCompression) {
    if (ShouldReportDetailedTime(r->ioptions.env, r->ioptions.stats)) {
      RecordTimeToHistogram(r->ioptions.stats, COMPRESSION_TIMES_NANOS,
                            timer.ElapsedNanos());
    }
    RecordInHistogram(r->ioptions.stats, BYTES_COMPRESSED,
                      raw_block_contents.size());
    RecordTick(r->ioptions.stats, NUMBER_BLOCK_COMPRESSED);
  } else if (*type != r->compression_type) {
    RecordTick(r->ioptions.stats, NUMBER_BLOCK_NOT_COMPRESSED);
  }
}

}  // namespace ROCKSDB_NAMESPACE