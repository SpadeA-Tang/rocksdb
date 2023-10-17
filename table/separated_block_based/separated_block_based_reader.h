#pragma once

#include <cstdint>

#include "table/table_reader.h"

namespace ROCKSDB_NAMESPACE {

class TableReader;

class SeparatedBlockBasedTable : TableReader {
 public:
  // 1-byte compression type + 32-bit checksum
  static constexpr size_t kBlockTrailerSize = 5;
};

}  // namespace ROCKSDB_NAMESPACE