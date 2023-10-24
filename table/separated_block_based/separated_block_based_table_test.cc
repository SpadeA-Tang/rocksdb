#include "db/snapshot_impl.h"
#include "db/table_properties_collector.h"
#include "file/file_util.h"
#include "options/options_helper.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/db.h"
#include "rocksdb/file_system.h"
#include "separated_block_based_table_factory.h"
#include "table/block_based/partitioned_index_iterator.h"
#include "table/format.h"
#include "table/separated_block_based/separated_block_based_table_builder.h"
#include "table/separated_block_based/separated_block_based_table_reader.h"
#include "test_util/testharness.h"
#include "test_util/testutil.h"
#include "util/random.h"

namespace ROCKSDB_NAMESPACE {

class SeparatedBlockTest
    : public testing::Test,
      public testing::WithParamInterface<std::tuple<
          CompressionType, bool, BlockBasedTableOptions::IndexType, bool>> {

  class MvccComparatorImpl : Comparator {
   public:
    MvccComparatorImpl() {}
    static const char* kClassName() { return "MvccComparatorImpl"; }
    const char* Name() const override {return kClassName(); }

    int Compare(const Slice& a, const Slice& b) const override {
      assert(false);
    }

    bool Equal(const Slice& a, const Slice& b) const override { assert(false); }

    void FindShortestSeparator(std::string* start,
                               const Slice& limit) const override {
      assert(false);
    }

    void FindShortSuccessor(std::string* key) const override {
      assert(false);
    }

    bool IsSameLengthImmediateSuccessor(const Slice& s,
                                        const Slice& t) const override {
      assert(false);
    }

    bool CanKeysWithDifferentByteContentsBeEqual() const override {
      assert(false);
    }

    using Comparator::CompareWithoutTimestamp;
    int CompareWithoutTimestamp(const Slice& a, bool /*a_has_ts*/, const Slice& b,
                                bool /*b_has_ts*/) const override {
      assert(false);
    }

    bool EqualWithoutTimestamp(const Slice& a, const Slice& b) const override {
      assert(false);
    }
  };

 protected:
  static std::string ToInternalKey(const std::string& key, SequenceNumber s) {
    InternalKey internal_key(key, s, ValueType::kTypeValue);
    return internal_key.Encode().ToString();
  }

  CompressionType compression_type_;
  bool use_direct_reads_;

  void SetUp() override {
    test_dir_ = test::PerThreadDBPath("SeparatedBlockTest");
    env_ = Env::Default();
    fs_ = FileSystem::Default();
    ASSERT_OK(fs_->CreateDir(test_dir_, IOOptions(), nullptr));

    BlockBasedTableOptions opts;
    opts.index_type = BlockBasedTableOptions::IndexType::kBinarySearch;
    opts.no_block_cache = false;
    opts.block_size = 600;
    opts.block_restart_interval = 4;
    table_factory_.reset(static_cast<SeparatedBlockBasedTableFactory*>(
        NewSeparatedBlockBasedTableFactory(opts)));
  }

  void TearDown() override { EXPECT_OK(DestroyDir(env_, test_dir_)); }

  std::string internal_key(uint32_t key, SequenceNumber seq) {
    char k[9] = {0};
    sprintf(k, "%08u", key);
    return SeparatedBlockTest::ToInternalKey(k, seq);
  }

  void CreateTableWithDefaultData(const std::string& table_name, uint32_t num, uint32_t seq_nums) {
    std::vector<std::pair<std::string, std::string>> kv;
    {
      Random rnd(101);
        for (uint32_t key = 0; key < num; key++) {
          std::string v;
          for (uint32_t ts = seq_nums; ts > 0; ts--) {
            std::string ikey = internal_key(key, SequenceNumber{ts});
            v = rnd.HumanReadableString(ts+5);
            kv.emplace_back(ikey, v);
          }
      }
    }
    CreateTable(table_name, CompressionType::kNoCompression, kv);
  }

  void CreateTable(const std::string& table_name,
                   const CompressionType& compression_type,
                   const std::vector<std::pair<std::string, std::string>>& kv) {
    std::unique_ptr<WritableFileWriter> writer;
    NewFileWriter(table_name, &writer);

    // Create table builder.
    Options options;
    ImmutableOptions ioptions(options);
    InternalKeyComparator comparator(options.comparator);
    ColumnFamilyOptions cf_options;
    MutableCFOptions moptions(cf_options);
    IntTblPropCollectorFactories factories;
    std::unique_ptr<TableBuilder> table_builder(table_factory_->NewTableBuilder(
        TableBuilderOptions(ioptions, moptions, comparator, &factories,
                            compression_type, CompressionOptions(),
                            0 /* column_family_id */, kDefaultColumnFamilyName,
                            -1 /* level */),
        writer.get()));

    // Build table.
    for (auto it = kv.begin(); it != kv.end(); it++) {
      table_builder->Add(it->first, it->second);
    }
    ASSERT_OK(table_builder->Finish());
  }

  void NewSeparatedBlockBasedTableReader(
      const FileOptions& foptions, const ImmutableOptions& ioptions,
      const InternalKeyComparator& comparator, const std::string& table_name,
      std::unique_ptr<SeparatedBlockBasedTable>* table) {
    std::unique_ptr<RandomAccessFileReader> file;
    NewFileReader(table_name, foptions, &file);

    uint64_t file_size = 0;
    ASSERT_OK(env_->GetFileSize(Path(table_name), &file_size));

    std::unique_ptr<TableReader> table_reader;
    ReadOptions ro;
    const auto* table_options =
        table_factory_->GetOptions<BlockBasedTableOptions>();
    ASSERT_NE(table_options, nullptr);
    ASSERT_OK(SeparatedBlockBasedTable::Open(
        ro, ioptions, EnvOptions(), *table_options, comparator, std::move(file),
        file_size, &table_reader));

    table->reset(reinterpret_cast<SeparatedBlockBasedTable*>(table_reader.release()));
  }

  std::string Path(const std::string& fname) { return test_dir_ + "/" + fname; }

 private:
  std::string test_dir_;
  Env* env_;
  std::shared_ptr<FileSystem> fs_;
  std::unique_ptr<SeparatedBlockBasedTableFactory> table_factory_;

  void NewFileWriter(const std::string& filename,
                     std::unique_ptr<WritableFileWriter>* writer) {
    std::string path = Path(filename);
    EnvOptions env_options;
    FileOptions foptions;
    std::unique_ptr<FSWritableFile> file;
    ASSERT_OK(fs_->NewWritableFile(path, foptions, &file, nullptr));
    writer->reset(new WritableFileWriter(std::move(file), path, env_options));
  }

  void NewFileReader(const std::string& filename, const FileOptions& opt,
                     std::unique_ptr<RandomAccessFileReader>* reader) {
    std::string path = Path(filename);
    std::unique_ptr<FSRandomAccessFile> f;
    ASSERT_OK(fs_->NewRandomAccessFile(path, opt, &f, nullptr));
    reader->reset(new RandomAccessFileReader(std::move(f), path,
                                             env_->GetSystemClock().get()));
  }
};

TEST_F(SeparatedBlockTest, TestBuilder) {
  CreateTableWithDefaultData("test", 100, 3);
  std::unique_ptr<SeparatedBlockBasedTable> table;
  Options options;
  ImmutableOptions ioptions(options);
  FileOptions foptions;
  foptions.use_direct_reads = use_direct_reads_;
  InternalKeyComparator comparator(options.comparator);
  NewSeparatedBlockBasedTableReader(foptions, ioptions, comparator, "test",
                                    &table);

  // Full Scan
  {
    ReadOptions read_options;
    read_options.all_versions = true;
    SnapshotImpl s;
    s.number_ = 3;
    read_options.snapshot = &s;
    const MutableCFOptions moptions(options);
    std::unique_ptr<InternalIterator> table_iter(
        table->NewIterator(read_options, moptions.prefix_extractor.get(),
                           nullptr, false, TableReaderCaller::kUncategorized));
    table_iter->SeekToFirst();
    uint32_t verify = 0;
    while (table_iter->Valid()) {
      char k[9] = {0};
      sprintf(k, "%08u", verify);

      Slice key(table_iter->key());
      ParsedInternalKey ikey;
      ParseInternalKey(key, &ikey, false);
      assert(ikey.user_key.compare(Slice{k}) == 0);
      assert(ikey.sequence == SequenceNumber{3});
      table_iter->Next();

      key = table_iter->key();
      ParseInternalKey(key, &ikey, false);
      assert(ikey.user_key.compare(Slice{k}) == 0);
      assert(ikey.sequence == SequenceNumber{2});
      table_iter->Next();

      key = table_iter->key();
      ParseInternalKey(key, &ikey, false);
      assert(ikey.user_key.compare(Slice{k}) == 0);
      assert(ikey.sequence == SequenceNumber{1});
      table_iter->Next();
      verify++;
    }
    assert(verify == 100);
  }

  // not scan all version, can read latest
  {
    ReadOptions read_options;
    read_options.all_versions = false;
    SnapshotImpl s;
    s.number_ = 3;
    read_options.snapshot = &s;
    const MutableCFOptions moptions(options);
    std::unique_ptr<InternalIterator> table_iter(
        table->NewIterator(read_options, moptions.prefix_extractor.get(),
                           nullptr, false, TableReaderCaller::kUncategorized));
    table_iter->SeekToFirst();
    uint32_t verify = 0;
    while (table_iter->Valid()) {
      char k[9] = {0};
      sprintf(k, "%08u", verify);

      Slice key(table_iter->key());
      ParsedInternalKey ikey;
      ParseInternalKey(key, &ikey, false);
      assert(ikey.user_key.compare(Slice{k}) == 0);
      assert(ikey.sequence == SequenceNumber{3});
      table_iter->Next();

      verify++;
    }
    assert(verify == 100);
  }

  // not scan all version, (the second latest)
  {
    ReadOptions read_options;
    read_options.all_versions = false;
    SnapshotImpl s;
    s.number_ = 2;
    read_options.snapshot = &s;
    const MutableCFOptions moptions(options);
    std::unique_ptr<InternalIterator> table_iter(
        table->NewIterator(read_options, moptions.prefix_extractor.get(),
                           nullptr, false, TableReaderCaller::kUncategorized));
    table_iter->SeekToFirst();
    uint32_t verify = 0;
    while (table_iter->Valid()) {
      char k[9] = {0};
      sprintf(k, "%08u", verify);

      Slice key(table_iter->key());
      ParsedInternalKey ikey;
      ParseInternalKey(key, &ikey, false);
      assert(ikey.user_key.compare(Slice{k}) == 0);
      assert(ikey.sequence == SequenceNumber{2});
      table_iter->Next();

      verify++;
    }
    assert(verify == 100);
  }

  {
    ReadOptions read_options;
    read_options.all_versions = false;
    SnapshotImpl s;
    s.number_ = 3;
    read_options.snapshot = &s;
    const MutableCFOptions moptions(options);
    std::unique_ptr<InternalIterator> table_iter(
        table->NewIterator(read_options, moptions.prefix_extractor.get(),
                           nullptr, false, TableReaderCaller::kUncategorized));
    table_iter->Seek(internal_key(53, 2));
    while (table_iter->Valid()) {
      Slice key(table_iter->key());
      ParsedInternalKey ikey;
      ParseInternalKey(key, &ikey, false);
      Slice value(table_iter->value());
      std::cout << "key " << ikey.user_key.ToString() << ", sequence " << ikey.sequence
                << ", value " << value.ToString() << std::endl;
      table_iter->Next();
    }
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}