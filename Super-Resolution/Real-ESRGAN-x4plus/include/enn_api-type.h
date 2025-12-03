
/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung
 * Electronics.
 */

/**
 * @file enn_api-type.h
 * @author Hoon Choi (hoon98.choi@)
 * @brief type definitions for public header files of enn API
 * @version 0.1
 * @date 2020-12-28
 */

#ifndef SRC_CLIENT_INCLUDE_ENN_API_TYPE_H_
#define SRC_CLIENT_INCLUDE_ENN_API_TYPE_H_

#include <stdint.h>

#define ENN_INFO_GRAPH_STR_LENGTH_MAX (256)
#define ENN_SHAPE_CHANNEL_MAX (10)
#define ENN_SHAPE_NAME_MAX (100)
#define ENN_C_STR_BUFFER_MAX (512)  // size of char* buffer cannot be larger than this

typedef uint64_t EnnModelId;  // higher 32-bits are initialized with zero
typedef uint64_t EnnExecuteModelId;  // higher 32-bits are initialized with zero
const EnnExecuteModelId EXEC_MODEL_NOT_ASSIGNED = 0;

typedef int32_t enn_preset_id;  // preset ID
#ifdef __LP64__
typedef unsigned long addr_t;
#else
typedef unsigned int addr_t;
#endif

typedef enum _EnnReturn {
    ENN_RET_SUCCESS = 0,
    ENN_RET_FAILED,
    ENN_RET_IO,
    ENN_RET_INVAL,
    ENN_RET_FILTERED,
    ENN_RET_CANCELED,
    ENN_RET_MEM_ERR,
    ENN_RET_SIZE,
    ENN_RET_FAILED_TIMEOUT_ENN = 10,
    ENN_RET_FAILED_TIMEOUT_DD,
    ENN_RET_FAILED_TIMEOUT_FW,
    ENN_RET_FAILED_TIMEOUT_HW_NOTRECOVERED,
    ENN_RET_FAILED_TIMEOUT_HW_RECOVERED,
    ENN_RET_FAILED_SERVICE_NULL,
    ENN_RET_FAILED_RESOURCE_BUSY,
    ENN_RET_NOT_SUPPORTED = 0xFF,
} EnnReturn;

/* NOTE: should be sync with types.hal */
typedef enum _enn_buf_dir_e {
  ENN_DIR_IN,
  ENN_DIR_OUT,
  ENN_DIR_EXT,
  ENN_DIR_NONE,
  ENN_DIR_SIZE
} enn_buf_dir_e;

// data structure for user buffer
typedef struct _ennBuffer {
    void *va;
    uint32_t size;  // requested size
    uint32_t offset;
} EnnBuffer;

typedef EnnBuffer* EnnBufferPtr;

typedef struct _NumberOfBuffersInfo {
    uint32_t n_in_buf;
    uint32_t n_out_buf;
    //  uint32_t n_ext_buf;  // not provided
} NumberOfBuffersInfo;

// Callback function prototype
typedef void (*EnnCallbackFunctionPtr)(addr_t *, addr_t);

typedef struct _ennBufferInfo {
    bool     is_able_to_update;
    uint32_t n;  // batch size
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint32_t size;
    uint32_t buffer_type;
    const char *label;
} EnnBufferInfo;

typedef struct _ennBufferMetaData {
    const char *label;
    uint32_t size;
    uint32_t data_type;
    uint32_t data_layout;
    uint32_t shape_num;
    uint32_t shape[10];
} EnnBufferMetaData;

enum TensorType : int8_t {
    TensorType_FLOAT32 = 0,
    TensorType_FLOAT16 = 1,
    TensorType_INT32 = 2,
    TensorType_UINT8 = 3,
    TensorType_INT64 = 4,
    TensorType_STRING = 5,
    TensorType_BOOL = 6,
    TensorType_INT16 = 7,
    TensorType_COMPLEX64 = 8,
    TensorType_INT8 = 9,
    TensorType_FLOAT64 = 10,
    TensorType_COMPLEX128 = 11,
    TensorType_UINT64 = 12,
    TensorType_RESOURCE = 13,
    TensorType_VARIANT = 14,
    TensorType_UINT32 = 15,
    TensorType_MIN = TensorType_FLOAT32,
    TensorType_MAX = TensorType_UINT32
};

enum DataLayout : int8_t {
    DataLayout_NCHW = 0,
    DataLayout_NHWC = 1,
    DataLayout_CELL = 2,
    DataLayout_UNDEFINE = 100,
    DataLayout_MIN = DataLayout_NCHW,
    DataLayout_MAX = DataLayout_UNDEFINE
};

typedef EnnBuffer** EnnBufferSet;

typedef struct _ennBatchInferenceBuffer {
    void *va;
    uint32_t size;
} EnnBatchInferenceBuffer;

/* Per Batch buffer sets. */
typedef struct _ennBatchBuffers {
    EnnBatchInferenceBuffer *inputs;  // buffer array for an inference
    EnnBatchInferenceBuffer *outputs;  // buffer array for an inference
} EnnBatchBuffers;

typedef struct _ennBatchBufferInfo {
    EnnBatchBuffers *batch_buffers;
} EnnBatchBufferInfo;

typedef enum _enn_meta_type_id_e {
    // NOTE(hoon98.choi): Core checks the identifer is in [Compiler ~ Dsp_fw]
    // Information from ENN framework (client)
    ENN_META_VERSION_FRAMEWORK = 110,
    ENN_META_VERSION_COMMIT = 111,
    ENN_META_VERSION_NNC_SCHEMA = 112,
    // Information from loaded model
    ENN_META_VERSION_MODEL_COMPILER_NNC = 120,
    ENN_META_VERSION_MODEL_COMPILER_NPU = 121,
    ENN_META_VERSION_MODEL_COMPILER_DSP = 122,
    ENN_META_VERSION_MODEL_SCHEMA = 123,
    ENN_META_VERSION_MODEL_VERSION = 124,
    // Information from ENN framework (service, dd)
    ENN_META_VERSION_DD = 130,
    ENN_META_VERSION_UNIFIED_FW = 131,
    ENN_META_VERSION_NPU_FW = 132,    // optional
    ENN_META_VERSION_DSP_FW = 133,
    ENN_META_VERSION_NCP = 134,
} EnnMetaTypeId;

typedef enum _PerfModePreference {
  ENN_PREF_MODE_LOW_POWER = 0,
  ENN_PREF_MODE_BALANCED = 4,
  ENN_PREF_MODE_PERFORMANCE = 7,
  ENN_PREF_MODE_ADAPTIVE = 10,
  ENN_PREF_MODE_BOOST = 1,
  ENN_PREF_MODE_CUSTOM = 5,
  ENN_PREF_MODE_RESERVED_1 = 101,
  ENN_PREF_MODE_RESERVED_2 = 102,
  ENN_PREF_MODE_RESERVED_3 = 103,

  ENN_PREF_MODE_NORMAL = 0,                 // This mode will be deprecated and is replaced with low power mode.
  ENN_PREF_MODE_BOOST_ON_EXE = 2,           // This mode will be deprecated and removed.
  ENN_PREF_MODE_BOOST_ON_OPEN_EXE = 4,      // This mode will be deprecated and is replaced with balanced mode.
  ENN_PREF_MODE_BOOST_ON_EXE_MAX = 6,       // This mode will be deprecated and removed.
  ENN_PREF_MODE_BOOST_ON_OPEN_EXE_MAX = 7,  // This mode will be deprecated and is replaced with performance mode.
  ENN_PREF_MODE_BOOST_ON_OPEN = 8,          // This mode will be deprecated and removed.
  ENN_PREF_MODE_BOOST_ON_OPEN_MAX = 9,      // This mode will be deprecated and removed.
} PerfModePreference;

typedef struct _ennModelPreference {
    uint32_t           preset_id;
    PerfModePreference pref_mode;  // Default preset for the model
} EnnModelPreference;

#endif  // SRC_CLIENT_INCLUDE_ENN_API_TYPE_H_
