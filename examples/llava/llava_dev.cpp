#include "clip.h"
#include "llava-utils.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void show_additional_info(int /*argc*/, char ** argv) {
    printf("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    printf("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

int main(int argc, char ** argv) {
    ggml_time_init();
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }

    const char * clip_path = params.mmproj.c_str();
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);

    llama_backend_init(params.numa);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;
    model_params.main_gpu = params.main_gpu;
    model_params.tensor_split = params.tensor_split;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx < 2048 ? 2048 : params.n_ctx;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    ctx_params.seed = params.seed;

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);


    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    while (true) {
        std::string input;
        std::cout << "Enter the path to a new image (or type 'exit' to quit): ";
        std::cin >> input;

        if (input == "exit") {
            break;
        }

        params.image = input;
        const char * img_path = params.image.c_str();

        clip_image_u8 img;
        clip_image_f32 img_res;

        if (!clip_image_load_from_file(img_path, &img)) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, img_path);
            continue;  // Skip to the next iteration
        }

        if (!clip_image_preprocess(ctx_clip, &img, &img_res, /*pad2square =*/ true)) {
            fprintf(stderr, "%s: unable to preprocess %s\n", __func__, img_path);
            continue;  // Skip to the next iteration
        }

        // processes each image and generates text here)
        // Don't free ctx_clip or ctx_llama inside this loop

        int n_img_pos  = clip_n_patches(ctx_clip);
        int n_img_embd = clip_n_mmproj_embd(ctx_clip);

        float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip));

        if (!image_embd) {
            fprintf(stderr, "Unable to allocate memory for image embeddings\n");
            continue;  // Skip to the next iteration
        }

        const int64_t t_img_enc_start_us = ggml_time_us();
        if (!clip_image_encode(ctx_clip, params.n_threads, &img_res, image_embd)) {
            fprintf(stderr, "Unable to encode image\n");
            continue;  // Skip to the next iteration
        }
        const int64_t t_img_enc_end_us = ggml_time_us();

        

        // make sure that the correct mmproj was used, i.e., compare apples to apples
        const int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));

        if (n_img_embd != n_llama_embd) {
            printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_img_embd, n_llama_embd);

            llama_free(ctx_llama);
            llama_free_model(model);
            llama_backend_free();
            free(image_embd);

            return 1;
        }

        // process the prompt
        // llava chat format is "<system_prompt>USER: <image_embeddings>\n<textual_prompt>\nASSISTANT:"

        int n_past = 0;

        const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;

        eval_string(ctx_llama, "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:", params.n_batch, &n_past, true);
        eval_image_embd(ctx_llama, image_embd, n_img_pos, params.n_batch, &n_past);
        eval_string(ctx_llama, (params.prompt + "\nASSISTANT:").c_str(), params.n_batch, &n_past, false);

        // generate the response

        printf("\n");
        printf("prompt: '%s'\n", params.prompt.c_str());
        printf("\n");

        for (int i = 0; i < max_tgt_len; i++) {
            const char * tmp = sample(ctx_llama, params, &n_past);
            if (strcmp(tmp, "</s>") == 0) break;

            printf("%s", tmp);
            fflush(stdout);
        }

        printf("\n");

        {
            const float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

            printf("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / n_img_pos);
        }

        llama_print_timings(ctx_llama);

        // free image_embd
        free(image_embd);


        // Before freeing ctx_llama
        ctx_llama->logits.clear();
        ctx_llama->logits.shrink_to_fit();
        ctx_llama->embedding.clear();
        ctx_llama->embedding.shrink_to_fit();
        ctx_llama->work_buffer.clear();
        ctx_llama->work_buffer.shrink_to_fit();
        // ...
        free(ctx_llama);
        llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    }

    // Cleanup after exiting the loop
    clip_free(ctx_clip);
    llama_free(ctx_llama);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
