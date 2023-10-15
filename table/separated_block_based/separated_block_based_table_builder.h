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
  struct Rep;
  Rep* rep_;
};



}  // namespace ROCKSDB_NAMESPACE