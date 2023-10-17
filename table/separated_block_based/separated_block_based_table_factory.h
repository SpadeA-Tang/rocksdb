#pragma once
#include <stdint.h>

#include <memory>
#include <string>

#include "port/port.h"
#include "rocksdb/flush_block_policy.h"
#include "rocksdb/table.h"

namespace ROCKSDB_NAMESPACE {
struct ColumnFamilyOptions;
struct ConfigOptions;
struct DBOptions;
struct EnvOptions;

class BlockBasedTableBuilder;
class RandomAccessFileReader;
class WritableFileWriter;

class SeparatedBlockBasedTableFactory : public TableFactory {
 public:
  explicit SeparatedBlockBasedTableFactory(
      const BlockBasedTableOptions& table_options = BlockBasedTableOptions());

  ~SeparatedBlockBasedTableFactory() {}

  // Method to allow CheckedCast to work for this class
  static const char* kClassName() { return kBlockBasedTableName(); }

  const char* Name() const override { return kBlockBasedTableName(); }

  using TableFactory::NewTableReader;
  Status NewTableReader(
      const ReadOptions& ro, const TableReaderOptions& table_reader_options,
      std::unique_ptr<RandomAccessFileReader>&& file, uint64_t file_size,
      std::unique_ptr<TableReader>* table_reader,
      bool prefetch_index_and_filter_in_cache = true) const override;

  TableBuilder* NewTableBuilder(
      const TableBuilderOptions& table_builder_options,
      WritableFileWriter* file) const override;

  void InitializeOptions();

 private:
  BlockBasedTableOptions table_options_;
};

}  // namespace ROCKSDB_NAMESPACE