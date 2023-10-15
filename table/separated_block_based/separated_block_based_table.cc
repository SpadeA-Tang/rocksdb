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

#include "rocksdb/cache.h"
#include "rocksdb/comparator.h"
#include "rocksdb/env.h"
#include "rocksdb/filter_policy.h"
#include "rocksdb/flush_block_policy.h"
#include "rocksdb/merge_operator.h"
#include "rocksdb/table.h"
#include "rocksdb/types.h"
#include "table/format.h"
#include "table/meta_blocks.h"
#include "table/table_builder.h"
#include "util/coding.h"
#include "util/compression.h"
#include "util/stop_watch.h"
#include "util/string_util.h"
#include "util/work_queue.h"

namespace ROCKSDB_NAMESPACE {

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

struct SeparatedBlockBasedTableBuilder::Rep {
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
  const Slice* first_key_in_next_old_block = nullptr;
  std::vector<std::unique_ptr<CompressionContext>> compression_ctxs;
  std::vector<std::unique_ptr<UncompressionContext>> verify_ctxs;

  TableProperties props;
  
  enum class State {
    kBuffered,
    kUnbuffered,
    kClosed,
  };
  State state;

  BlockHandle pending_handle;  // Handle to add to index block

  std::unique_ptr<FilterBlockBuilder> filter_builder;

  std::string compressed_output;
  std::unique_ptr<FlushBlockPolicy> flush_block_policy;
  std::unique_ptr<FlushBlockPolicy> flush_old_block_policy;

 private:
  // Synchronize status & io_status accesses across threads from main thread,
  // compression thread and write thread in parallel compression.
  std::mutex status_mutex;
  std::atomic<bool> status_ok;
  Status status;
};

}  // namespace ROCKSDB_NAMESPACE