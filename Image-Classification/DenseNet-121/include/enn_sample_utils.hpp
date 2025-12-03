#pragma once

#include <iostream>
#include <map>

#define RED         "\x1b[31m"
#define GREEN         "\x1b[32m"
#define YELLOW         "\x1b[33m"
#define BLUE         "\x1b[34m"
#define MAGENTA         "\x1b[35m"
#define CYAN         "\x1b[36m"
#define RESET         "\x1b[0m"

#define TEST_MODEL_PATH_DENSENET "/data/vendor/densenet/densenet121_simplify_O2_MultiCore.nnc"
#define TEST_INPUT_FILE_PATH "/data/vendor/densenet/media/image.jpg"

extern bool g_enable_log; 

#define PRINT_ERROR(message, ...) \
    if (g_enable_log) { \
        printf(RED "TEST: " message, ##__VA_ARGS__); \
        printf("\n" RESET); \
    }

#define PRINT(message, ...) \
    if (g_enable_log) { \
        printf("TEST: " message, ##__VA_ARGS__); \
        printf("\n"); \
    }

namespace enn {
namespace sample_utils {
int get_file_size(const char *filename) {
    FILE *f = fopen(filename, "rb");

    if (f == nullptr) {
        std::cerr << "File open Error!: " << filename << std::endl;
        return -1;
    }

    fseek(f, 0, SEEK_END);
    int size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fclose(f);

    return size;
}

int import_file_to_mem(const char *filename, char *target_va) {
    auto file_size = get_file_size(filename);
    if (file_size < 0) {
        std::cerr << "Wrong file size!: " << file_size << std::endl;
        return -1;
    }

    FILE *f = fopen(filename, "rb");

    if (file_size != static_cast<int>(fread(target_va, sizeof(char), file_size, f))) {
        std::cerr << "File fread Error!: " << filename << ", size: " << file_size << std::endl;
        return -1;
    }

    fclose(f);

    return file_size;
}

int export_mem_to_file(const char *filename, const void *va, uint32_t size) {
    size_t ret_cnt;

    PRINT("DEBUG:: Export memory to file: name(%s) va(%p), size(%d)", filename, va, size);

    FILE *fp = fopen(filename, "wb");

    ret_cnt = fwrite(va, size, 1, fp);
    if (ret_cnt <= 0) {
        PRINT("FileWrite Failed!!(%zu)", ret_cnt);
        fclose(fp);
        return ENN_RET_INVAL;
    }

    PRINT("DEBUG:: File Save Completed.");
    fclose(fp);

    return ENN_RET_SUCCESS;
}

void show_raw_memory_to_hex(uint8_t *va, uint32_t size, const int line_max, const int size_max) {
    char line_tmp[100] = {0,};
    int int_size = static_cast<int>(size);
    int max = (size_max == 0 ? int_size : (int_size < size_max ? int_size : size_max));
    int idx = sprintf(line_tmp, "[%p] ", va);  // prefix of line
    int i = 0;                                 // idx records current location of print line
    for (; i < max; ++i) {
        idx += sprintf(&(line_tmp[idx]), "%02X ", va[i]);
        if (i % line_max == (line_max - 1)) {
            // if new line is required, flush print --> and record prefix print
            line_tmp[idx] = 0;
            std::cout << line_tmp << std::endl;
            idx = 0;
            idx = sprintf(line_tmp, "[%p] ", &(va[i]));
        }
    }
    if (i % line_max != 0) {
        std::cout << line_tmp << std::endl;
    }
}

void print_buffer_hex(uint8_t *ptr, const int length, int line_max)
{
    printf( " [ %s , %d, %d] ", __func__, length, line_max);

    for(int i = 0 ; i < length ; i++)
    {
        if( i == 0 || (i%line_max) == 0 )
        {
            printf("\n[%p] index(%4d) -" , ptr, i);
        }
        printf(" %02x,", *ptr);
        ptr++;
    }
    printf("\n " );
}

void print_buffer_info(EnnBufferInfo info_p, const enn_buf_dir_e direction)
{
    if(direction == ENN_DIR_IN)
    {
        printf(YELLOW);
        printf( " >>>>>>>>>>>> Input Buffer information  \n" );
        printf( " is_able_to_update : %d \n" , info_p.is_able_to_update);
        printf( " batch size        : %d \n" , info_p.n);
        printf( " width             : %d \n" , info_p.width);
        printf( " height            : %d \n" , info_p.height);
        printf( " channel           : %d \n" , info_p.channel);
        printf( " size              : %d \n" , info_p.size);
        printf( " buffer type       : %d \n" , info_p.buffer_type);
        printf( " label             : %s \n" , info_p.label);
        printf("\n" RESET);
    }
    else if(direction == ENN_DIR_OUT)
    {
        printf(YELLOW);
        printf( " <<<<<<<<<<<< Output Buffer information  \n" );
        printf( " is_able_to_update : %d \n" , info_p.is_able_to_update);
        printf( " batch size        : %d \n" , info_p.n);
        printf( " width             : %d \n" , info_p.width);
        printf( " height            : %d \n" , info_p.height);
        printf( " channel           : %d \n" , info_p.channel);
        printf( " size              : %d \n" , info_p.size);
        printf( " buffer type       : %d \n" , info_p.buffer_type);
        printf( " label             : %s \n" , info_p.label);
        printf("\n" RESET);
    }
    return;
}

class DmaAllocator {
    std::map<int, EnnBufferPtr> bufs;

 public:
    DmaAllocator() {
        bufs.clear();
    }

    ~DmaAllocator() {
        for (auto& buf : bufs) {
            enn::api::EnnReleaseBuffer(buf.second);
        }
    }

    /**
     * @brief Allocate DMA buffer and return its fd.
     *        Although this function uses EnnCreateBuffer for allocating dma buffer,
     *        other common methods is available too.
     * @param req_size   size of dma buffer
     * @param buf   EnnBufferPtr to store dam buffer
     * @return int fd of dam buffer
     */
    int allocate_dma_buffer(const uint32_t req_size) {
        int fd;
        EnnBufferPtr buf;
        if (enn::api::EnnCreateBuffer(&buf, req_size) != ENN_RET_SUCCESS) {
            PRINT("# allocate dma_buffer failed\n");
            return -1;
        }

        if (enn::api::EnnGetFileDescriptorFromEnnBuffer(buf, &fd) != ENN_RET_SUCCESS) {
            PRINT("# allocate dma_buffer failed\n");
            enn::api::EnnReleaseBuffer(buf);
            return -1;
        }
        bufs[fd] = buf;

        return fd;
    }

    /**
     * @brief Release DMA buffer that allocated by 'allocate_dma_buffer' function.
     *        Although this function uses EnnCreateBuffer for allocating dma buffer,
     *        other common methods is available too.
     * @param buf   EnnBufferPtr to store dam buffer
     */
    void release_dma_buffer(const int fd) {
        auto buf = bufs[fd];
        enn::api::EnnReleaseBuffer(buf);
    }
};

void prepare_user_batch_buffers(size_t buffer_size,  uint32_t num_batch, uint32_t num_buffer,
                    EnnBatchInferenceBuffer *buffer, bool CHUNK_MEMORY_FLAG) {
    for (uint32_t i = 0; i < num_buffer; ++i) {
        /* heap memory allocated by user */
        void *base_va = reinterpret_cast<char *>(calloc(1, buffer_size * num_batch));
        /* For Single set case */
        if (CHUNK_MEMORY_FLAG) {
            buffer[i].va = base_va;
            buffer[i].size = buffer_size * num_batch;
        }
        for (uint32_t j = 0; j < num_batch; ++j) {
            uint8_t *va = static_cast<uint8_t*>(base_va) + buffer_size * j;

            /* For N set case */
            if (CHUNK_MEMORY_FLAG == false) {
                buffer[j*num_buffer+i].va = va;
                buffer[j*num_buffer+i].size = buffer_size;
            }
        }
    }
};

}  // namespace sample_utils
}  // namespace enn
