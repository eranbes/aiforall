// index.js — lesson registry. Order is the course order: top of the stack first.

import plan from './00-plan.js';
import tour from './00b-tour.js';
import sampling from './01-sampling.js';
import tokens from './02-tokens.js';
import softmax from './03-softmax.js';
import logits from './04-logits.js';
import blocks from './05-blocks.js';
import attention from './06-attention.js';
import mlp from './07-mlp.js';
import kernels from './08-kernels.js';
import quant from './09-quant.js';
import file from './10-file.js';
import training from './11-training.js';
import batching from './12-batching.js';
import caching from './13-caching.js';
import speculative from './14-speculative.js';
import streaming from './15-streaming.js';
import anatomy from './16-qwen-anatomy.js';
import deltanet from './17-deltanet.js';
import live from './18-live.js';
import behaviour from './19-behaviour.js';
import tools from './20-tools.js';
import context from './21-context.js';
import diffusion from './22-diffusion.js';

export const LESSONS = [plan, tour, sampling, tokens, softmax, logits, blocks, attention, mlp, kernels, quant, file, training, batching, caching, speculative, streaming, anatomy, deltanet, live, behaviour, tools, context, diffusion];
