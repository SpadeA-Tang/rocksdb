#pragma once
#include <stdint.h>

#include <string>

#include "rocksdb/table.h"
#include "table/block_based/block_builder.h"
#include "table/block_based/block_type.h"
#include "table/table_builder.h"

namespace ROCKSDB_NAMESPACE {

class SeparatedBlockBasedTableBuilder : public TableBuilder {
  SeparatedBlockBasedTableBuilder(
      const BlockBasedTableOptions& table_options,
      const TableBuilderOptions& table_builder_options,
      const UserComparatorWrapper& user_comparator, WritableFileWriter* file);

  // No copying allowed
  SeparatedBlockBasedTableBuilder(const SeparatedBlockBasedTableBuilder&) =
      delete;
  SeparatedBlockBasedTableBuilder& operator=(
      const SeparatedBlockBasedTableBuilder&) = delete;

  // REQUIRES: Either Finish() or Abandon() has been called.
  ~SeparatedBlockBasedTableBuilder();

  void Add(const Slice& key, const Slice& value) override;

  // Return non-ok iff some error has been detected.
  Status status() const override;

  // Return non-ok iff some error happens during IO.
  IOStatus io_status() const override;

  Status Finish() override;

 private:
  bool ok() const { return status().ok(); }

  struct Rep;
  Rep* rep_;

  struct ParallelCompressionRep;

  void WriteBlock(BlockBuilder* block, BlockHandle* handle, BlockType blocktype,
                  std::string* buffer);
  void WriteBlock(const Slice& block_contents, BlockHandle* handle,
                  BlockType block_type, std::string* buffer);
  void WriteRawBlock(const Slice& data, CompressionType, BlockHandle* handle,
                     BlockType block_type, const Slice* raw_data = nullptr,
                     bool is_top_level_filter_block = false);

  template <typename TBlocklike>
  Status InsertBlockInCache(const Slice& block_contents,
                            const BlockHandle* handle, BlockType block_type);

  Status InsertBlockInCacheHelper(const Slice& block_contents,
                                  const BlockHandle* handle,
                                  BlockType block_type,
                                  bool is_top_level_filter_block);

  Status InsertBlockInCompressedCache(const Slice& block_contents,
                                      const CompressionType type,
                                      const BlockHandle* handle);

  void CompressAndVerifyBlock(const Slice& raw_block_contents,
                              bool is_data_block,
                              const CompressionContext& compression_ctx,
                              UncompressionContext* verify_ctx,
                              std::string* compressed_output,
                              Slice* result_block_contents,
                              CompressionType* result_compression_type,
                              Status* out_status);

  void FlushNewDataBlock();
  void FlushOldDataBlock();

  const uint64_t kCompressionSizeLimit = std::numeric_limits<int>::max();
};

}  // namespace ROCKSDB_NAMESPACE