#pragma once
#include "table/separated_block_based/index_reader_common.h"
#include "table/separated_block_based/separated_block_based_table_reader.h"

namespace ROCKSDB_NAMESPACE {
// Index that allows binary search lookup for the first key of each block.
// This class can be viewed as a thin wrapper for `Block` class which already
// supports binary search.
class SeparatedBinarySearchIndexReader : public SeparatedBlockBasedTable::IndexReaderCommon {
 public:
  // Read index from the file and create an intance for
  // `BinarySearchIndexReader`.
  // On success, index_reader will be populated; otherwise it will remain
  // unmodified.
  static Status Create(const rocksdb::SeparatedBlockBasedTable* table,
                const ReadOptions& ro, FilePrefetchBuffer* prefetch_buffer,
                bool use_cache, bool prefetch, bool pin,
                BlockCacheLookupContext* lookup_context,
                std::unique_ptr<IndexReader>* index_reader,
                std::unique_ptr<IndexReader>* old_index_reader);

  InternalIteratorBase<IndexValue>* NewIterator(
      const ReadOptions& read_options, bool /* disable_prefix_seek */,
      IndexBlockIter* iter, GetContext* get_context,
      BlockCacheLookupContext* lookup_context) override;

  size_t ApproximateMemoryUsage() const override {
    size_t usage = ApproximateIndexBlockMemoryUsage();
#ifdef ROCKSDB_MALLOC_USABLE_SIZE
    usage += malloc_usable_size(const_cast<SeparatedBinarySearchIndexReader*>(this));
#else
    usage += sizeof(*this);
#endif  // ROCKSDB_MALLOC_USABLE_SIZE
    return usage;
  }

 private:
  SeparatedBinarySearchIndexReader(const SeparatedBlockBasedTable* t,
                          CachableEntry<Block>&& index_block)
      : IndexReaderCommon(t, std::move(index_block)) {}
};
}  // namespace ROCKSDB_NAMESPACE