#include "table/separated_block_based/separated_block_based_table_iterator.h"

namespace ROCKSDB_NAMESPACE {
void SeparatedBlockBasedTableIterator::Seek(const Slice& target) {
  SeekImpl(&target);
}

void SeparatedBlockBasedTableIterator::SeekToFirst() { SeekImpl(nullptr); }

void SeparatedBlockBasedTableIterator::SeekImpl(const rocksdb::Slice* target) {
  is_out_of_bound_ = false;
  is_at_first_key_from_index_ = false;
  if (target && !CheckPrefixMayMatch(*target, IterDirection::kForward)) {
    ResetDataIter();
    return;
  }

  bool need_seek_index = true;
  if (block_iter_points_to_real_block_ && block_iter_.Valid()) {
  }

  if (need_seek_index) {
    if (target) {
      index_iter_->Seek(*target);
      if (!latest_only) {
        old_index_iter_->Seek(*target);
      }
    } else {
      index_iter_->SeekToFirst();
      if (!latest_only) {
        old_index_iter_->SeekToFirst();
      }
    }

    if (!index_iter_->Valid()) {
      ResetDataIter();
    }
  }

  IndexValue v = index_iter_->value();
  const bool same_block = block_iter_points_to_real_block_ &&
                          v.handle.offset() == prev_block_offset_;
}
}  // namespace ROCKSDB_NAMESPACE