#pragma once
#include <stdint.h>

#include <string>

#include "rocksdb/table.h"

namespace ROCKSDB_NAMESPACE {

class SeparatedBlockBasedTableBuilder : public TableBuilder {
  SeparatedBlockBasedTableBuilder(
      const BlockBasedTableOptions& table_options,
      const TableBuilderOptions& table_builder_options,
      WritableFileWriter* file);

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

  void WriteBlock(BlockBuilder* block, BlockHandle* handle,
                  BlockType blocktype,std::string* buffer));
  void WriteBlock(const Slice& block_contents, BlockHandle* handle,
                  BlockType block_type, std::string* buffer);
  void WriteRawBlock(const Slice& data, CompressionType, BlockHandle* handle,
                     BlockType block_type, const Slice* raw_data = nullptr,
                     bool is_top_level_filter_block = false);

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
};



}  // namespace ROCKSDB_NAMESPACE