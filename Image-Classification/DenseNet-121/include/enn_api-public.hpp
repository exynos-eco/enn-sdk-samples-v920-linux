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
 * @file enn_api-public.hpp
 * @version ENN_API_VERSION_HPP 2.0
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @date 2021-10-15
 */

/**
 * @mainpage Introduction
 * @section ENN Framework API
 *
 * public header file for enn API supported by C++ syntax
 *
 * 1. Basically, user can simply execute model with this call flow:
 * ~~~~~~~~~~~~~~~~~~~~~
 *      EnnInitialize() ->
 *      EnnOpenModel() ->
 *      EnnAllocateAllBuffers() ->
 *      (copy input to input buffer) ->
 *      EnnExecuteModel() ->
 *      (check output from output buffer) ->
 *      EnnReleaseBuffers() ->
 *      EnnCloseModel() ->
 *      EnnDeinitialize()
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * 2. User can manually set each memory buffer as this flow:
 * ~~~~~~~~~~~~~~~~~~~~~
 *      EnnInitialize() ->
 *      EnnOpenModel() ->
 *      (Get Buffer Information with EnnGetBuffersInfo(), EnnGetBufferInfoByXXX()) ->
 *      (Create In/Out Buffers with EnnCreateBuffer(), EnnCreateBufferFromFd()) ->
 *      (Set Buffer for commit with EnnSetBufferByIndex(), EnnSetBufferByLabel()) ->
 *      EnnBufferCommit() ->
 *      EnnExecuteModel() ->
 *      (Release Buffers with EnnReleaseBuffer()) ->
 *      EnnCloseModel() ->
 *      EnnDeinitialize()
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 *
 * 3. User can set multiple set of execution buffer as flow:
 * ( With these APIs, the user manages buffer set with circular queue )
 * ~~~~~~~~~~~~~~~~~~~~~
 *      EnnInitialize() ->
 *      EnnOpenModel() ->
 *      (Get Buffer Information with EnnGetBuffersInfo(), EnnGetBufferInfoByXXX()) ->
 *      (Create Multiple set of Buffers with EnnCreateBuffer(), EnnCreateBufferFromFd()) ->
 *      (Set Buffer for commit to multiple session ID with EnnSetBufferByIndex(), EnnSetBufferByLabel()) ->
 *      EnnBufferCommit(...., session_id) ->
 *      EnnExecuteModel(model_id, session_id) ->
 *      (Release Buffers with EnnReleaseBuffer()) ->
 *      EnnCloseModel() ->
 *      EnnDeinitialize()
 * ~~~~~~~~~~~~~~~~~~~~~
 * > User can manage buffers pipelined and ping-pong
 * >   with import ion(dmabufheap) allocated buffer from outside
 * >   and multiple buffer spaces without memory copies
 *
 * 4. A caller also can call preference, security, version related APIs
 * ~~~~~~~~~~~~~~~~~~~~~
 *      EnnGetMetaInfo()
 *      EnnSet/GetPreferenceXXX()
 *      EnnSecureXXXX()
 *       :
 *       :
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * 5. User can uncommit and unset buffers to save resources when the model is not used for a long time.
 * (Uncommit / Unset APIs are optional because EnnCloseModel() has all these functionalities)
 * ~~~~~~~~~~~~~~~~~~~~~
 *      EnnInitialize() ->
 *      EnnOpenModel() ->
 *      EnnAllocateAllBuffers() ->
 *      (copy input to input buffer) ->
 *      EnnExecuteModel() ->
 *      (check output from output buffer) ->
 *      EnnBufferUncommit() ->
 *      EnnUnsetBuffers() ->
 *      EnnReleaseBuffers() ->
 *      (User can reuse model with new buffers) ->
 *      EnnCloseModel() ->
 *      EnnDeinitialize()
 * ~~~~~~~~~~~~~~~~~~~~~
 *
 * @version 0.1
 * @date 2021-06-03, 2021-09-03
 * @note This file wraps enn_api-public.h
 *
 */

#pragma once

#include "enn_api-type.h"

namespace enn {
namespace api {

/**
 * @defgroup api_context Context initialize / deinitialize
 */

/**
 * @brief Initialize Enn Framework.
 * Framework generates context in a caller's process.
 * Context counts the number of initialize/deinitialize pair.
 *
 * @ingroup api_context
 * @return EnnReturn result
 */
EnnReturn EnnInitialize(void);

/**
 * @brief Deinitialize Enn Framework.
 * Framework degenerates context in a caller's process.
 *
 * @ingroup api_context
 * @return EnnReturn result
 */
EnnReturn EnnDeinitialize(void);

/**
 * @defgroup api_openmodel OpenModel / CloseModel related
 */

/**
 * @brief Open a Model from model file
 *
 * @param model_file [IN] model_file, output from graph-gen. A caller must have access to this file.
 * @param model_id [OUT] model_id, 64 bit unsigned int
 * @ingroup api_openmodel
 * @return EnnReturn result
 */
EnnReturn EnnOpenModel(const char *model_file, EnnModelId *model_id);

/**
 * @brief Open a Model from memory
 *
 * @param va [IN] memory address which a model loaded from
 * @param size [IN] size of the memory
 * @param model_id [OUT] model_id, 64 bit unsigned int
 * @ingroup api_openmodel
 * @return EnnReturn result
 */
EnnReturn EnnOpenModelFromMemory(const char *va, const uint32_t size, EnnModelId *model_id);


/**
 * @brief Open Model with {fd, size} and preference
 *
 * @param _fd [IN] file descriptor from ion(dmabufheap) buffer includes output from graph-gen (ex .nnc)
 * @param model_id [OUT] model_id, 64 bit unsigned int
 * @ingroup api_openmodel
 * @return EnnReturn result
 */
EnnReturn EnnOpenModelFromFd(int _fd, EnnModelId *model_id);

/**
 * @brief Open a NNC Model from model file with separate weight buffers
 *
 * @param model_file [IN] NNC model_file, output from graph-gen. A caller must have access to this file.
 * @param weights [IN] EnnBuffer array which contains weight buffers
 * @param n_weights [IN] number of weight buffers
 * @param model_id [OUT] model_id, 64 bit unsigned int
 * @ingroup api_openmodel
 * @return EnnReturn result
 */
EnnReturn EnnOpenModelWithWeight(const char *model_file, EnnBufferPtr* weights, uint32_t n_weights, EnnModelId *model_id);

/**
 * @brief Open a NNC Model from memory with separate weight buffers
 *
 * @param va [IN] memory address which a NNC model loaded from
 * @param size [IN] size of the memory
 * @param weights [IN] EnnBuffer array which contains weight buffers
 * @param n_weights [IN] number of weight buffers
 * @param model_id [OUT] model_id, 64 bit unsigned int
 * @ingroup api_openmodel
 * @return EnnReturn result
 */
EnnReturn EnnOpenModelFromMemoryWithWeight(const char *va, const uint32_t size,
                                           EnnBufferPtr* weights, uint32_t n_weights, EnnModelId *model_id);

/**
 * @brief Open a model using a file descriptor.
 *
 * @param[in] fd File descriptor
 * @param[in] size Model size
 * @param[in] offset File offset
 * @param[out] model_id Model ID
 *
 * @return ENN_RETKPI_SUCCESS if successful, otherwise an error code
 */
EnnReturn EnnOpenModelWithFileOpenFd(const int fd, const uint32_t size, const uint32_t offset, EnnModelId *model_id);

/**
 * @brief Open a model using a file descriptor with EnnBuffers contain weights
 *
 * @param[in] fd File descriptor
 * @param[in] size Model size
 * @param[in] offset File offset
 * @param[in] weights Array of weights
 * @param[in] n_weights Size of the weights array
 * @param[out] model_id Model ID
 *
 * @return ENN_RET_SUCCESS if successful, otherwise an error code
 */
EnnReturn EnnOpenModelWithFileOpenFdWeight(const int fd, const uint32_t size, const uint32_t offset,
                                                  EnnBufferPtr *weights, uint32_t n_weights, EnnModelId *model_id);


/**
 * @brief Close a Model and Free all resources generated by EnnOpenModel()
 *
 * @param model_id [IN] model_id generated from EnnOpenModel()
 * @ingroup api_openmodel
 * @return EnnReturn result
 */
EnnReturn EnnCloseModel(const EnnModelId model_id);

/**
 * @defgroup api_memory Memory Handling
 */

/**
 * @brief Create EnnBuffer with requested size
 * This API support ION or dmabufheap which can be used in a device(DSP/CPU/NPU/GPU)
 *
 * @param out [OUT] output buffer pointer. User can get va, size, offset through *out
 * @param req_size [IN] request size
 * @param is_cached [IN] flag, the buffer uses cache or not
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnCreateBuffer(EnnBufferPtr *out, const uint32_t req_size, const bool is_cached = true);

/**
 * @brief Create EnnBuffer by importing the fd of the buffer allocated by ION/DMABUFHEAP
 *
 * @param out [OUT] output buffer pointer. User can get va, size, offset through *out
 * @param fd [IN] file descriptor generated from ION/DMABUFHEAP allocation
 * @param size [IN] size of buffer
 * @param offset [IN] if a caller wants to make buffer from a part of fd, use this.
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnCreateBufferFromFd(EnnBufferPtr *out, const uint32_t fd, const uint32_t size, const uint32_t offset = 0);

/**
 * @brief Get File Descriptor from EnnBuffer
 *
 * @param buffer [IN] EnnBuffer pointer
 * @param fd [OUT] available File Descriptor value, otherwise filled -1
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnGetFileDescriptorFromEnnBuffer(EnnBufferPtr buffer, int32_t *fd);

/**
 * @brief Allocate all buffers which a caller should allocate
 *
 * @param model_id [IN] model_id from EnnOpenModel()
 * @param out_buffers [OUT] pointer of EnnBuffer array
 * @param buf_info [OUT] size of the array
 * @param session_id [IN] after generate buffer space, user can set this field if session_id > 0
 * @param do_commit [IN] if true, the framework tries to commit after buffer allocation
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnAllocateAllBuffers(const EnnModelId model_id, EnnBufferPtr **out_buffers, NumberOfBuffersInfo *buf_info,
                                const int session_id = 0, const bool do_commit = true);

/**
 * @brief Release a buffer which generated by EnnCreateBuffer()
 *
 * @param buffer [IN] buffer object from EnnCreateBuffer()
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnReleaseBuffer(EnnBufferPtr buffer);

/**
 * @brief Release a buffer array which generated by EnnAllocatedAllBuffers()
 * This API includes releasing all elements in the array.
 *
 * @param buffers [IN] pointer of buffer array
 * @param numOfBuffers [IN] size of buffer array
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnReleaseBuffers(EnnBufferPtr *buffers, const int32_t numOfBuffers);

/**
 * @brief Create EnnBuffer with requested size with optional flag
 * This API support ION or dmabufheap which can be used in a device(DSP/CPU/NPU/GPU)
 *
 * @param out [OUT] output buffer pointer. User can get va, size, offset through *out
 * @param flag [IN] optional flag for ENN Framework
 * @param req_size [IN] request size
 * @param is_cached [IN] flag, the buffer uses cache or not
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnCreateBufferWithFlag(EnnBufferPtr *out, const uint32_t flag, const uint32_t req_size,
                                  const bool is_cached = true);

/**
 * @brief Create EnnBuffer by importing the fd of the buffer allocated by ION/DMABUFHEAP with optional flag
 *
 * @param out [OUT] output buffer pointer. User can get va, size, offset through *out
 * @param flag [IN] optional flag for ENN Framework
 * @param fd [IN] file descriptor generated from ION/DMABUFHEAP allocation
 * @param size [IN] size of buffer
 * @param offset [IN] if a caller wants to make buffer from a part of fd, use this.
 * @ingroup api_memory
 * @return EnnReturn result
 */
EnnReturn EnnCreateBufferFromFdWithFlag(EnnBufferPtr *out, const uint32_t flag,
                                        const uint32_t fd, const uint32_t size, const uint32_t offset = 0);


/**
 * @defgroup api_model Setters and Getters for model
 */

/**
 * @brief Get the number of buffers from loaded model
 *
 * @param buffers_info [OUT] number of in / out buffer which caller should commit.
 * @param model_id [IN] model id from EnnOpenModel()
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnGetBuffersInfo(NumberOfBuffersInfo *buffers_info, const EnnModelId model_id);

/**
 * @brief Get one buffer information from loaded model
 * ```
 * typedef struct _ennBufferInfo {
 *     bool     is_able_to_update;  // this is not used
 *     uint32_t n;
 *     uint32_t width;
 *     uint32_t height;
 *     uint32_t channel;
 *     uint32_t size;
 *     const char *label;
 * } EnnBufferInfo;
 * ```
 * a caller can identify a buffer as {DIR, Index} such as {IN, 0}
 *
 * @note direction, this parameter refer to ENUM value named enn_buf_dir_e
 * @param out_buf_info [OUT] output buffer information
 * @param model_id [IN] model ID from load_model
 * @param direction [IN] direction of buffer
 * @param index [IN] buffer's index number in model
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnGetBufferInfoByIndex(EnnBufferInfo *out_buf_info, const EnnModelId model_id, const enn_buf_dir_e direction,
                                  const uint32_t index);

/**
 * @brief Get one buffer information from loaded model
 * ```
 * typedef struct _ennBufferInfo {
 *     bool     is_able_to_update;  // this is not used
 *     uint32_t n;
 *     uint32_t width;
 *     uint32_t height;
 *     uint32_t channel;
 *     uint32_t size;
 *     const char *label;
 * } EnnBufferInfo;
 * ```
 * a caller can identify a buffer as {label} or {tensor name}
 *
 * @param out_buf_info [OUT] output buffer information
 * @param model_id [IN] model ID from load_model
 * @param label [IN] label. if .nnc includes redundant label, the framework returns information of the first founded tensor.
 * C-style string type.
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnGetBufferInfoByLabel(EnnBufferInfo *out_buf_info, const EnnModelId model_id, const char *label);

/**
 * @brief Get one buffer metadata from loaded model
 * ```
 * typedef struct _ennBufferMetaData {
 *     const char *label;
 *     uint32_t size;
 *     uint32_t data_type;
 *     uint32_t data_layout;
 *     uint32_t shape_num;
 *     uint32_t shape[10];
 * } EnnBufferMetaData;
 * ```
 * a caller can identify a buffer as {DIR, Index} such as {IN, 0}
 *
 * @note direction, this parameter refer to ENUM value named enn_buf_dir_e
 * @param out_buf_meta [OUT] output buffer metadata
 * @param model_id [IN] model ID from load_model
 * @param direction [IN] direction of buffer
 * @param index [IN] buffer's index number in model
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnGetBufferMetaByIndex(EnnBufferMetaData *out_buf_meta, const EnnModelId model_id, const enn_buf_dir_e direction,
                                  const uint32_t index);

/**
 * @brief Get one buffer metadata from loaded model
 * ```
 * typedef struct _ennBufferMetaData {
 *     const char *label;
 *     uint32_t size;
 *     uint32_t data_type;
 *     uint32_t data_layout;
 *     uint32_t shape_num;
 *     uint32_t shape[10];
 * } EnnBufferMetaData;
 * ```
 * a caller can identify a buffer as {label} or {tensor name}
 *
 * @param out_buf_meta [OUT] output buffer metadata
 * @param model_id [IN] model ID from load_model
 * @param label [IN] label. if .nnc includes redundant label, the framework returns information of the first founded tensor.
 * C-style string type.
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnGetBufferMetaByLabel(EnnBufferMetaData *out_buf_meta, const EnnModelId model_id, const char *label);

/**
 * @brief Set memory object to commit-space.
 * A user can generates buffer space to commit. (Basically the framework generates 16 spaces)
 * "Set Buffer" means a caller can put its memory object to it's space. "Commit" means send
 * memory-buffer set which can run opened model completely to service core.
 *
 * @note direction, this parameter refer to ENUM value named enn_buf_dir_e
 * @param model_id [IN] model ID from load_model
 * @param direction [IN] direction of buffer
 * @param index [IN] index number of buffer
 * @param buf [IN] memory object from EnnCreateBufferXXX()
 * @param session_id [IN] If a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnSetBufferByIndex(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index,
                              EnnBufferPtr buf, const int session_id = 0);

/**
 * @brief Set large memory object to commit-space.
 * A user can generates buffer space to commit. (Basically the framework generates 16 spaces)
 * "Set Buffer" means a caller can put its memory object to it's space. "Commit" means send
 * memory-buffer set which can run opened model completely to service core.
 * If user set a memory object with this function, every EnnExecuteModel call moves internal offset by step_size.
 * Internal offset will be reset when it is over buffer size.
 *
 * @note direction, this parameter refer to ENUM value named enn_buf_dir_e
 * @param model_id [IN] model ID from load_model
 * @param direction [IN] direction of buffer
 * @param index [IN] index number of buffer
 * @param buf [IN] memory object from EnnCreateBufferXXX()
 * @param step_size [IN] memory distance between execution
 * @param init_step [IN] initial step count where start offset is buf_offset + init_step * step_size
 * @param session_id [IN] If a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnSetBufferByIndexWithStep(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index,
                                      EnnBufferPtr buf, const uint32_t step_size, const uint32_t init_step,
                                      const int session_id = 0);

/**
 * @brief Set memory object to commit-space.
 * A user can generates buffer space to commit. (Basically the framework generates 16 spaces)
 * "Set Buffer" means a caller can put its memory object to it's space. "Commit" means send
 * memory-buffer set which can run opened model completely to service core.
 *
 * @param model_id [IN] model ID from load_model
 * @param label [IN] label. if .nnc includes redundant label, the framework returns information of the first founded tensor.
 * C-style string type.
 * @param buf [IN] memory object from EnnCreateBufferXXX()
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnSetBufferByLabel(const EnnModelId model_id, const char *label, EnnBufferPtr buf, const int session_id = 0);

/**
 * @brief Set large memory object to commit-space.
 * A user can generates buffer space to commit. (Basically the framework generates 16 spaces)
 * "Set Buffer" means a caller can put its memory object to it's space. "Commit" means send
 * memory-buffer set which can run opened model completely to service core.
 * If user set a memory object with this function, every EnnExecuteModel call moves internal offset by step_size.
 * Internal offset will be reset when it is over buffer size.
 *
 * @param model_id [IN] model ID from load_model
 * @param label [IN] label. if .nnc includes redundant label, the framework returns information of the first founded tensor.
 * C-style string type.
 * @param buf [IN] memory object from EnnCreateBufferXXX()
 * @param step_size [IN] memory distance between execution
 * @param init_step [IN] initial step count where start offset is buf_offset + init_step * step_size
 * @param session_id [IN] If a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnSetBufferByLabelWithStep(const EnnModelId model_id, const char *label,
                                      EnnBufferPtr buf, const uint32_t step_size, const uint32_t init_step,
                                      const int session_id = 0);

/**
 * @brief Set multiple buffers to buffer space once.
 * EnnBuffer list should have the order: [IN/0] [IN/1] ... [IN/n-1] [OUT/0] [OUT/1]... [OUT/n-1]  (Direction/index)
 *
 * @param model_id [IN] model ID from load_model
 * @param bufs [IN] memory object list
 * @param sum_io [IN] number of buffers. (number of input + number of output)
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnSetBuffers(const EnnModelId model_id, EnnBuffer **bufs, const int32_t sum_io, const int session_id = 0);

/**
 * @brief Unset buffers from buffer space with session id = *session id*
 *
 * @note This function is available since ENN Version 2.1.x. It is not supported by ENN Version 2.0.x
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnUnsetBuffers(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Update memory object to commit-space, by overwriting buffer set information.
 * EnnUpdateBuffer must be called after calling this API.
 *
 * @note This function is available since ENN Version 2.2.x. It is not supported by ENN Version 2.1.x
 * @note Direction, this parameter refer to ENUM value named enn_buf_dir_e
 * @param model_id [IN] model ID from load_model
 * @param direction [IN] direction of buffer
 * @param index [IN] index number of buffer
 * @param buf [IN] memory object from EnnCreateBufferXXX()
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnUpdateBufferInfoByIndex(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index,
                                    EnnBufferPtr buf, const int session_id = 0);

/**
 * @brief Update memory object to commit-space, by overwriting buffer set information.
 * EnnUpdateBuffer must be called after calling this API.
 *
 * @note This function is available since ENN Version 2.2.x. It is not supported by ENN Version 2.1.x
 * @param model_id [IN] model ID from load_model
 * @param label [IN] label. if .nnc includes redundant label, the framework returns information of the first founded tensor.
 * C-style string type.
 * @param buf [IN] memory object from EnnCreateBufferXXX()
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_model
 * @return EnnReturn result
 */
EnnReturn EnnUpdateBufferInfoByLabel(const EnnModelId model_id, const char *label, EnnBufferPtr buf,
                                    const int session_id = 0);

/**
 * @defgroup api_commit Commit Buffer
 */

/**
 * @brief Send buffer-set to service core. session_id indicates which buffer space should be sent.
 * The committed buffers are released if a caller calls EnnCloseModel()
 *
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_commit
 * @return EnnReturn result
 */
EnnReturn EnnBufferCommit(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Cancel buffer commit. Uncommitted buffer could be reused by calling EnnBufferCommit() again.
 * Uncommit is optional for reusing buffer. Normally, calling EnnCloseModel() is enough to release committed buffers.
 *
 * @note This function is available since ENN Version 2.1.x. It is not supported by ENN Version 2.0.x
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_commit
 * @return EnnReturn result
 */
EnnReturn EnnBufferUncommit(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Send buffer-set to service core to update buffers that committed before.
 * session_id indicates which buffer space should be sent.
 * The committed buffers are released if a caller calls EnnCloseModel()
 *
 * @note This function is available since ENN Version 2.2.x. It is not supported by ENN Version 2.1.x
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] if a caller generates 2 or more buffer space, session_id can be an identifier
 * @ingroup api_commit
 * @return EnnReturn result
 */
EnnReturn EnnUpdateBuffer(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Set this model as a batch mode so that Enn prepare batch resources
 *
 * @param model_id [IN] model ID from load_model
 * @param buffers_info [IN] number of in / out buffer.
 * @ingroup api_commit
 * @return EnnReturn result
 */
EnnReturn EnnPrepareBatch(const EnnModelId model_id, NumberOfBuffersInfo *buffers_info);

/**
 * @brief Update Batch info with given buffer information
 *
 * @param model_id [IN] model ID from load_model
 * @param batch_info [IN] batch buffer info (EnnBatchBufferInfo) filled by user.
 * @param num_batch [IN] number of inference in a batch
 * @param is_chunk [IN] bool flag to inform each ifm/ofm is chunk buffer for all batch inferences
 * @ingroup api_commit
 * @return EnnReturn result
 */
EnnReturn EnnUpdateBatchInfo(const EnnModelId model_id, const EnnBatchBufferInfo *batch_info, int num_batch,
                                    bool is_chunk = false);

/**
 * @brief Set execute-time offset to a specific buffer of a session.
 * The session must be committed before calling this function.
 * It overrides step-feature.
 *
 * @param model_id [IN] model ID from load_model
 * @param direction [IN] Direction (IN/OUT)
 * @param index [IN] index number of buffer
 * @param offset [IN] additional buffer offset. Committed buffer must have enough size.
 * @param session_id [IN] session id to set buffer
 * @return EnnReturn zero if successful
 */
EnnReturn EnnSetExecuteOffsetByIndex(const EnnModelId model_id, const enn_buf_dir_e direction,
                                     const uint32_t index, uint32_t offset, const int session_id = 0);

/**
 * @brief Set execute-time offset to a specific buffer of a session. It overrides step-feature.
 * The session must be committed before calling this function.
 * It overrides step-feature.
 *
 * @param model_id [IN] model ID from load_model
 * @param label [IN] string pointer
 * @param offset [IN] additional buffer offset. Committed buffer must have enough size.
 * @param session_id [IN] session id to set buffer
 * @return EnnReturn zero if successful
 */
EnnReturn EnnSetExecuteOffsetByLabel(const EnnModelId model_id, const char *label,
                                     const uint32_t offset, const int session_id = 0);

/**
 * @defgroup api_execute Execute Models
 */

/**
 * @brief Request to service core to execute model with committed buffers
 *
 * @note this function runs in block mode
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] session ID
 * @ingroup api_execute
 * @return EnnReturn result
 */
EnnReturn EnnExecuteModel(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Request to service core to execute model with committed buffers
 *
 * @note this function run batch image using 4 session
 * @param model_id [IN] model ID from load_model
 * @ingroup api_execute
 * @return EnnReturn result
 */
EnnReturn EnnExecuteModelBatch(const EnnModelId model_id);
/**
 * @brief Request to service core to execute model in background asynchronously
 *
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] session ID
 * @ingroup api_execute
 * @return EnnReturn result
 */
EnnReturn EnnExecuteModelAsync(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Wait result of calling EnnExecuteModelAsync()
 * If execution is finished, this function is returned immediately
 * otherwise, this function would be blocked until the execution finished
 *
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] session ID
 * @ingroup api_execute
 * @return EnnReturn result
 */
EnnReturn EnnExecuteModelWait(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Execute multiple sessions, session 0 ~ (session_count - 1), at once for NPU/DSP only model
 *        all target sessions must be commited before calling it
 *        When calling this function, "session_count" MUST NOT BE CHANGED once batch is executed
 *        It does not check step feature, only current offset is applied
 *
 * @note  EXPERIMENTAL FEATURE
 * @param model_id model ID to be executed.
 * @param session_count session count to execute.
 * @return EnnReturn ENN_RET_SUCCESS if successful, others for failed.
 */
EnnReturn EnnExecuteModelWithSessionBatch(const EnnModelId model_id, const int session_count = 8);

/**
 * @brief Request to cancel the execution. This function should be called in another thread.
 *
 * @note this function runs in block mode
 * @param model_id [IN] model ID from load_model
 * @param session_id [IN] session ID
 * @ingroup api_execute
 * @return EnnReturn result
 */
EnnReturn EnnExecuteModelCancel(const EnnModelId model_id, const int session_id = 0);

/**
 * @brief Request to cancel the session batch execution. This function should be called in another thread.
 *
 * @note this function runs in block mode
 * @param model_id [IN] model ID from load_model
 * @ingroup api_execute
 * @return EnnReturn result
 */
EnnReturn EnnExecuteModelWithSessionBatchCancel(const EnnModelId model_id);

/**
 * @defgroup api_miscellaneous Security, preference, get meta information..
 */

/**
 * @brief Try to set secure mode
 *
 * @param heap_size [IN] request heap size
 * @param secure_heap_addr [OUT] 64 bit address from secure driver
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSecureOpen(const uint32_t heap_size, uint64_t *secure_heap_addr);

/**
 * @brief Close secure mode
 *
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSecureClose(void);

/**
 * @brief Setting Preset ID for operation performance
 *
 * @param val [IN] value to set preset ID
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSetPreferencePresetId(const uint32_t val);

/**
 * @brief Setting PerfConfig ID for operation performance
 *
 * @param val [IN] value to set PerfConfig ID
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSetPreferencePerfConfigId(const uint32_t val);

/**
 * @brief Setting Performance Mode
 *
 * @param val [IN] value to set Performance Mode
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSetPreferencePerfMode(const uint32_t val);

/**
 * @brief Setting Preset ID for time out
 *
 * @note The parameter in this function is in seconds
 * @param val [IN] value to set time out
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSetPreferenceTimeOut(const uint32_t val);

/**
 * @brief Setting priority value for NPU
 *
 * @param val [IN] value to set NPU job priority
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSetPreferencePriority(const uint32_t val);

/**
 * @brief Setting affinity to set NPU core operation
 *
 * @param val [IN] value to set affinity
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnSetPreferenceCoreAffinity(const uint32_t val);

/**
 * @brief Get current information for Preset ID
 *
 * @param val_ptr [OUT] current value of Preset ID
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetPreferencePresetId(uint32_t *val_ptr);

/**
 * @brief Get current information for PerfConfig ID
 *
 * @param val_ptr [OUT] current value of PerfConfig ID
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetPreferencePerfConfigId(uint32_t *val_ptr);

/**
 * @brief Get current information for Performance Mode
 *
 * @param val_ptr [OUT] current value of Performance Mode
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetPreferencePerfMode(uint32_t *val_ptr);

/**
 * @brief Get current information for Time Out
 *
 * @note The parameter in this function is in seconds
 * @param val_ptr [OUT] current value of Time Out
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetPreferenceTimeOut(uint32_t *val_ptr);


/**
 * @brief Get current information for NPU Priority
 *
 * @param val_ptr [OUT] current value of NPU Priority
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetPreferencePriority(uint32_t *val_ptr);


/**
 * @brief Get current information for NPU Core affinity
 *
 * @param val_ptr [OUT] current value of NPU Core affinity
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetPreferenceCoreAffinity(uint32_t *val_ptr);

/**
 * @brief Reset the internal value of a preference
 *
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
extern EnnReturn EnnResetPreferenceAsDefault(void);

/**
 * @brief Get Session ID for DSP
 *
 * @param model_id [IN] model ID
 * @param out [OUT] session ID from DSP
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnDspGetSessionId(const EnnModelId model_id, int32_t *out);

/**
 * @brief Get Meta Information
 * This API provide information about loaded model as well as framework
 *
 * @note This API provide information according to info_id, It represent about EnnMetaTypeId
 * Also it is already defined as ENUM. Here is ENUM value for use
 * ```
 * ENN_META_VERSION_FRAMEWORK
 * ENN_META_VERSION_COMMIT
 * ENN_META_VERSION_MODEL_COMPILER_NNC
 * ENN_META_VERSION_MODEL_COMPILER_NPU
 * ENN_META_VERSION_MODEL_COMPILER_DSP
 * ENN_META_VERSION_MODEL_SCHEMA
 * ENN_META_VERSION_MODEL_VERSION
 * ENN_META_VERSION_DD
 * ENN_META_VERSION_UNIFIED_FW
 * ENN_META_VERSION_NPU_FW
 * ENN_META_VERSION_DSP_FW
 * ```
 * @param info_id [IN] EnnMetaType ID
 * @param model_id [IN] model ID
 * @param output_str [OUT] result string
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
EnnReturn EnnGetMetaInfo(const EnnMetaTypeId info_id, const EnnModelId model_id,
                         char output_str[ENN_INFO_GRAPH_STR_LENGTH_MAX]);

/**
 * @brief Turn on execution message print
 *
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
extern EnnReturn EnnSetExecMsgAlwaysOn();

/**
 * @brief Turn off execution message print
 *
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
extern EnnReturn EnnSetExecMsgAlwaysOff();

/**
 * @brief Set frequency of execution message print
 *
 * @param rate [IN] if rate is N, the execution message shows every {1, N+1, 2N+1..} times.
 * @ingroup api_miscellaneous
 * @return EnnReturn result
 */
extern EnnReturn EnnSetExecMsgRate(uint32_t rate);

/**
 * @brief Explicitly generate buffer space to commit. If user don't call this, the system automatically generates one space.
 * Currently n_set should be less then 16.
 * "Commit" means send memory-buffer set which can run opened model completely to service core.
 *
 * @note This function is deprecated. Because Framework initializes 16 sessions in default by D/D restrictions.
 * @param model_id [IN] model ID from load_model
 * @param n_set [IN] number of buffer spaces
 * @return EnnReturn result
 */
EnnReturn EnnGenerateBufferSpace(const EnnModelId model_id, const int n_set = 16);

/**
 * @brief Update Enn Model elements.
 *
 * @note This function is not used yet
 * @param action [IN] action
 * @param model_id [IN] model_id
 * @param target_type [IN] target_type
 * @param target_id [IN] target_id
 * @param key [IN] key
 * @param value [IN] value to change
 * @return EnnReturn result
 */
extern EnnReturn EnnExecuteCommand(const char *action, const char *model_id, const char *target_type,
                                   const char *target_id, const char *key, const char *value);

}  // namespace api
}  // namespace enn
