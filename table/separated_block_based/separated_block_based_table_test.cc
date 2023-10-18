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
 protected:
 protected:
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
    table_factory_.reset(static_cast<SeparatedBlockBasedTableFactory*>(
        NewSeparatedBlockBasedTableFactory(opts)));
  }

  void TearDown() override { EXPECT_OK(DestroyDir(env_, test_dir_)); }

  void CreateTable(const std::string& table_name,
                   const CompressionType& compression_type,
                   const std::map<std::string, std::string>& kv) {
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
      for (uint64_t j = 3; j > 0; j--) {
        std::string k = ToInternalKey(it->first, SequenceNumber(j));
        std::string v = it->second;
        table_builder->Add(k, v);
      }
    }
    ASSERT_OK(table_builder->Finish());
  }

  void NewSeparatedBlockBasedTableReader(
      const FileOptions& foptions, const ImmutableOptions& ioptions,
      const InternalKeyComparator& comparator, const std::string& table_name,
      std::unique_ptr<BlockBasedTable>* table) {
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

    table->reset(reinterpret_cast<BlockBasedTable*>(table_reader.release()));
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

  static std::string ToInternalKey(const std::string& key, SequenceNumber s) {
    InternalKey internal_key(key, s, ValueType::kTypeValue);
    return internal_key.Encode().ToString();
  }
};

TEST_F(SeparatedBlockTest, TestBuilder) {
  std::map<std::string, std::string> kv;
  {
    Random rnd(101);
    uint32_t key = 0;
    for (int block = 0; block < 10; block++) {
      for (int i = 0; i < 16; i++) {
        char k[9] = {0};
        // Internal key is constructed directly from this key,
        // and internal key size is required to be >= 8 bytes,
        // so use %08u as the format string.
        sprintf(k, "%08u", key);
        std::string v;
        v = rnd.HumanReadableString(10);
        kv[std::string(k)] = v;
        key++;
      }
    }
  }

  CreateTable("test", CompressionType::kNoCompression, kv);
  std::unique_ptr<BlockBasedTable> table;
  Options options;
  ImmutableOptions ioptions(options);
  FileOptions foptions;
  foptions.use_direct_reads = use_direct_reads_;
  InternalKeyComparator comparator(options.comparator);
  NewSeparatedBlockBasedTableReader(foptions, ioptions, comparator, "test",
                                    &table);
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}