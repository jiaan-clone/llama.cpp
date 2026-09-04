// 引入 llama.cpp 的公共 C API，包括模型、上下文、分词、解码和采样接口。
#include "llama.h"
// 引入区域设置接口，供 std::setlocale 和 LC_NUMERIC 使用。
#include <clocale>
// 引入 C 标准输入输出接口，供 printf、fprintf 和 fflush 使用。
#include <cstdio>
// 引入 C 字符串接口，供 strcmp 使用。
#include <cstring>
// 引入 C++ 字符串类型 std::string。
#include <string>
// 引入 C++ 动态数组类型 std::vector。
#include <vector>

// 程序入口；argc 是参数数量，argv 是参数字符串数组。
int main(int argc, char ** argv) {
    // 固定数字区域设置为 C，确保浮点数始终使用小数点，避免本地化格式影响解析或输出。
    std::setlocale(LC_NUMERIC, "C");

    // 保存 GGUF 模型文件路径，初始为空，后续必须由 -m 参数提供。
    std::string model_path="/home/king/llama.cpp/deepseek-r1-1.5b-f16.gguf";
    // 保存生成文本所基于的提示词，并设置默认提示词。
    std::string prompt = "Hello my name is";
    // 保存要卸载到 GPU 的模型层数，默认尝试卸载 99 层。
    // 卸载到 GPU 的模型层数”指的是：把模型中的多少个 Transformer 层的权重和计算任务放到 GPU 上执行。
    
    int ngl = 0;
    // 保存最多要生成的 token 数量，默认生成 4 个。
    int n_predict = 4;

    // 动态加载并注册所有可用的 GGML 计算后端，例如 CPU、CUDA 或其他已构建后端。
    ggml_backend_load_all();

    // 创建一份模型加载参数，并用 llama.cpp 提供的默认值初始化。
    llama_model_params model_params = llama_model_default_params();
    // 设置尽可能卸载到 GPU 的模型层数。
    model_params.n_gpu_layers = ngl;

    // 这里提到模型和参数。模型：真正读取 的GGUF 文件中权重、词表等数据。
    // 模型参数 model_params ，指 加载配置，model 是按照该配置加载出来的实际模型。
    // 根据文件路径和加载参数读取 GGUF 模型；失败时返回空指针。
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    // 检查模型是否加载成功。
    if (model == NULL) {
        // 向标准错误输出函数名和模型加载失败信息。
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        // 返回非零值，终止程序。
        return 1;
    }

    // 从模型取得词表的只读指针；词表的生命周期由模型管理。
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 第一次调用不提供输出缓冲区，用负返回值取得提示词所需的 token 数量。
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // 按计算出的 token 数量分配数组，用于保存提示词的全部 token ID。
    std::vector<llama_token> prompt_tokens(n_prompt);
    // 第二次调用执行实际分词；两个 true 分别表示添加特殊 token 和解析文本中的特殊 token。
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        // 分词失败时向标准错误输出错误信息。
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        // 返回非零值，终止程序。
        return 1;
    }

    // 创建一份推理上下文参数，并用 llama.cpp 提供的默认值初始化。
    llama_context_params ctx_params = llama_context_default_params();
    // 将上下文容量设为提示词 token 数加生成 token 数；减一是因为循环按当前批次预留位置。
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // 将单次解码可处理的最大 token 数设为提示词长度，使整个提示词可一次送入模型。
    ctx_params.n_batch = n_prompt;
    // 不禁用性能统计，即启用上下文内部的性能计数器。
    ctx_params.no_perf = false;

    // 使用已加载模型和上下文参数创建推理上下文；失败时返回空指针。
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    // 检查推理上下文是否创建成功。
    if (ctx == NULL) {
        // 向标准错误输出函数名和上下文创建失败信息。
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        // 返回非零值，终止程序。
        return 1;
    }

    // 创建采样器链参数，并用默认值初始化。采样器链（Sampler Chain） 是一组按顺序执行的 token 选择规则，用于根据模型输出的 logits 决定下一个 token。
    auto sparams = llama_sampler_chain_default_params();
    // 不禁用性能统计，即启用采样器链的性能计数器。
    sparams.no_perf = false;
    // 根据参数创建采样器链，用于组合一个或多个采样策略。
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // 向采样器链加入贪心采样器，每一步都选择概率最高的 token。
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // 逐个遍历提示词中的 token ID，以原始文本形式打印提示词。
    for (auto id : prompt_tokens) {
        // 创建固定长度的字符缓冲区，用于接收当前 token 对应的文本片段。
        char buf[128];
        // 将 token ID 转换为文本片段；最后一个 true 表示输出特殊 token 的可打印形式。
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        // 负返回值表示转换失败或缓冲区不足。
        if (n < 0) {
            // 向标准错误输出 token 转文本失败信息。
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            // 返回非零值，终止程序。
            return 1;
        }
        // 使用实际返回长度构造字符串，不依赖缓冲区以空字符结尾。
        std::string s(buf, n);
        // 将当前 token 对应的文本片段输出到标准输出。
        printf("%s", s.c_str());
    }

    // 将提示词 token 数组包装成一个无需额外分配的解码批次。
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // 记录主解码循环开始时的微秒时间戳。
    const auto t_main_start = ggml_time_us();
    // 记录已经生成并输出的 token 数量。
    int n_decode = 0;
    // 声明变量，用于保存每次采样得到的新 token ID。
    llama_token new_token_id;

    // 在提示词和目标生成长度允许的范围内持续执行解码；n_pos 表示已经处理的位置数。
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // 用 Transformer 模型计算当前批次，并把结果及 KV 缓存写入上下文。
        // 推理上下文 ctx 与 kv 缓存的关系：ctx 是一次推理的运行环境，KV 缓存是 ctx 内部管理的重要数据。
        /**
         * 创建 ctx 时，会根据 ctx_params.n_ctx 等参数准备 KV 缓存空间。
         * 每次调用 llama_decode(ctx, batch); 模型会做如下操作：
         * 1. 读取之前 token 的 KV 缓存；
         * 2. 计算当前 token；
         * 3. 把当前 token 的 Key/Value 写入缓存。
         * 这样生成后续token 时，就不用重新计算整个提示词。
         * 因此，ctx 是完整的推理状态容器，KV缓存是其中用于加速注意力计算的数据；n_ctx 决定 KV 缓存最多能保存多长的上下文；
         * llama_free(ctx) 会释放上下文及其管理的 KV 缓存
         */ 
        // 第一次 batch 包含整个提示词，模型一次处理所有提示词 token。
        // 之后每个 batch 通常只包含上一步生成的一个 token。
        if (llama_decode(ctx, batch)) {
            // 解码失败时向标准错误输出错误信息和返回码。
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            // 返回非零值，终止程序。
            return 1;
        }

        // 将当前位置推进当前批次包含的 token 数量。
        n_pos += batch.n_tokens;

        // 使用当前解码结果生成下一个 token。
        {
            // 从上下文最后一组 logits（索引 -1）中按采样器链规则选出下一个 token。
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // 判断新 token 是否为 EOG（生成结束）类型的特殊 token。
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                // 模型表示生成结束时跳出主循环。
                break;
            }

            // 创建固定长度的字符缓冲区，用于接收新 token 对应的文本片段。
            char buf[128];
            // 将新 token ID 转换为可输出的文本片段。
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            // 负返回值表示转换失败或缓冲区不足。
            if (n < 0) {
                // 向标准错误输出 token 转文本失败信息。
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                // 返回非零值，终止程序。
                return 1;
            }
            // 使用实际返回长度构造字符串，不依赖缓冲区以空字符结尾。
            std::string s(buf, n);
            // 将新生成的文本片段输出到标准输出。
            printf("%s", s.c_str());
            // 立即刷新标准输出，使生成内容可以逐 token 显示。
            fflush(stdout);

            // 把刚采样出的 token 包装成下一轮要解码的单 token 批次。
            batch = llama_batch_get_one(&new_token_id, 1);

            // 将已生成 token 计数加一。
            n_decode += 1;
        }
    }

    // 在生成文本末尾输出换行。
    printf("\n");

    // 记录主解码循环结束时的微秒时间戳。
    const auto t_main_end = ggml_time_us();

    // 输出生成 token 数、总耗时和平均每秒生成 token 数。
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            // 传入函数名、token 数、由微秒换算的秒数，以及 token 数除以耗时得到的速度。
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    // 向标准错误输出一个空行，分隔性能信息。
    fprintf(stderr, "\n");
    // 输出采样器链收集的性能统计。
    llama_perf_sampler_print(smpl);
    // 输出推理上下文收集的编码、提示词处理和解码性能统计。
    llama_perf_context_print(ctx);
    // 再输出一个空行，使日志结尾更清晰。
    fprintf(stderr, "\n");

    // 释放采样器链及其中包含的采样器。
    llama_sampler_free(smpl);
    // 释放推理上下文及其 KV 缓存等资源。
    llama_free(ctx);
    // 释放已加载的模型资源。
    llama_model_free(model);

    // 返回 0，表示程序正常结束。
    return 0;
}
