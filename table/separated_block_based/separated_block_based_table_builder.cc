#include "table/separated_block_based/separated_block_based_table_builder.h"

#include <assert.h>
#include <stdio.h>

#include <atomic>
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
#include "table/block_based/block_based_table_builder.h"
#include "table/block_based/block_based_table_reader.h"
#include "table/block_based/block_like_traits.h"
#include "table/block_based/filter_block.h"
#include "table/block_based/filter_policy_internal.h"
#include "table/block_based/index_builder.h"
#include "table/block_based/parsed_full_filter_block.h"
#include "table/format.h"
#include "table/meta_blocks.h"
#include "table/separated_block_based/separated_block_based_reader.h"
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
}


struct SeparatedBlockBasedTableBuilder::Rep {
  Rep(const BlockBasedTableOptions& table_opt, const TableBuilderOptions& tbo, const UserComparatorWrapper& user_comparator_,
      WritableFileWriter* f):ioptions(tbo.ioptions),
                               moptions(tbo.moptions),table_options(table_opt), user_comparator(user_comparator_),internal_comparator(tbo.internal_comparator),file(f),offset(0),old_offset(0),alignment(table_options.block_align
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
                            flush_block_policy(
                                table_options.flush_block_policy_factory->NewFlushBlockPolicy(
                                    table_options, data_block)),
                            flush_old_block_policy(
                                table_options.flush_block_policy_factory->NewFlushBlockPolicy(
                                    table_options, data_block)),
                            reason(tbo.reason),
                            status_ok(true),
                            io_status_ok(true) 
  {

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

  uint64_t get_offset() { return offset.load(std::memory_order_relaxed); }
  void set_offset(uint64_t o) { offset.store(o, std::memory_order_relaxed); }

  uint64_t get_old_offset() {
    return old_offset.load(std::memory_order_relaxed);
  }
  void set_old_offset(uint64_t o) {
    old_offset.store(o, std::memory_order_relaxed);
  }

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
  const UserComparatorWrapper& user_comparator;
  const InternalKeyComparator& internal_comparator;
  WritableFileWriter* file;
  std::atomic<uint64_t> offset;
  std::atomic<uint64_t> old_offset;
  size_t alignment;
  BlockBuilder data_block;
  BlockBuilder old_data_block;
  std::vector<BlockHandle> old_block_handles;
  std::string old_data_buffer;

  std::unique_ptr<IndexBuilder> index_builder;

  std::string last_key;
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
    const BlockBasedTableOptions& table_options,
    const TableBuilderOptions& table_builder_options,
    WritableFileWriter* file) {
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

  rep_ = new Rep(sanitized_table_options, tbo, file);

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
        r->user_comparator.CompareWithoutTimestamp(key, r->last_key) != 0) {
      auto should_flush = r->flush_block_policy->Update(key, value);
      if (should_flush) {
        assert(!r->data_block.empty());
        r->first_key_in_next_block = &key;
        FlushNewDataBlock();
        // todo: kBuffered

        if (ok() && r->state == Rep::State::kUnbuffered) {
            r->index_builder->AddIndexEntry(&r->last_key, &key,
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
        size_t ts_sz =
              r->internal_comparator.user_comparator()->timestamp_size();
        r->filter_builder->Add(ExtractUserKeyAndStripTimestamp(key, ts_sz));
      }

      r->data_block.AddWithLastKey(key, value, r->last_key);
    } else {
      r->old_data_block.AddWithLastKey(key, value, r->last_key);
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
  WriteBlock(&r->data_block, &r->pending_handle, BlockType::kData, nullptr);
}

void SeparatedBlockBasedTableBuilder::FlushOldDataBlock() {
  Rep* r = rep_;
  assert(rep_->state != Rep::State::kClosed);
  if (!ok()) return;
  if (r->data_block.empty()) return;
  r->old_block_handles.push_back({});
  WriteBlock(&r->old_data_block, &(r->old_block_handles.back()),
             BlockType::kData, &r->old_data_buffer);
}

void SeparatedBlockBasedTableBuilder::WriteBlock(BlockBuilder* block,
                                                 BlockHandle* handle,
                                                 BlockType block_type,
                                                 std::string* buffer) {
  block->Finish();
  std::string raw_block_contents;
  raw_block_contents.reserve(rep_->table_options.block_size);
  block->SwapAndReset(raw_block_contents);
  // todo: kBuffered
  WriteBlock(raw_block_contents, handle, block_type, buffer);
}

void SeparatedBlockBasedTableBuilder::WriteBlock(
    const Slice& raw_block_contents, BlockHandle* handle, BlockType block_type,
    std::string* buffer) {
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

  if (buffer == nullptr) {
    WriteRawBlock(block_contents, type, handle, block_type,
                  &raw_block_contents);
  } else {
    handle->set_offset(r->get_old_offset());
    handle->set_size(block_contents.size());
    // todo: checksum
  }
  r->compressed_output.clear();
  if (is_data_block) {
    if (r->filter_builder != nullptr) {
      r->filter_builder->StartBlock(r->get_offset());
    }
    r->props.data_size = r->get_offset();
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
    r->file->InvalidateCache(static_cast<size_t>(r->get_offset()), size)
        .PermitUncheckedError();
  }
  return s;
}

void SeparatedBlockBasedTableBuilder::WriteRawBlock(
    const Slice& block_contents, CompressionType type, BlockHandle* handle,
    BlockType block_type, const Slice* raw_block_contents,
    bool is_top_level_filter_block) {
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