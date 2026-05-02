#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <arm_neon.h>

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(const string &input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits += 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}

static inline bit32 LoadLE32(const Byte *p)
{
	return ((bit32)p[0]) |
		   ((bit32)p[1] << 8) |
		   ((bit32)p[2] << 16) |
		   ((bit32)p[3] << 24);
}

static inline bit32 ByteSwap32(bit32 value)
{
	return ((value & 0xff) << 24) |
		   ((value & 0xff00) << 8) |
		   ((value & 0xff0000) >> 8) |
		   ((value & 0xff000000) >> 24);
}

// 用已经完成预处理的 paddedMessage 计算串行 MD5，避免回退时重复 StringProcess
static void MD5HashFromPadded(const Byte *paddedMessage, int messageLength, bit32 *state)
{
	int n_blocks = messageLength / 64;

	// bit32* state= new bit32[4];
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];

		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = LoadLE32(paddedMessage + 4 * i1 + i * 64);
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x2441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x4881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		state[i] = ByteSwap32(state[i]);
	}
}

// NEON SIMD版本的MD5基本逻辑函数
#define F_SIMD(x, y, z) \
	vorrq_u32(vandq_u32((x), (y)), vandq_u32(vmvnq_u32((x)), (z)))

#define G_SIMD(x, y, z) \
	vorrq_u32(vandq_u32((x), (z)), vandq_u32((y), vmvnq_u32((z))))

#define H_SIMD(x, y, z) \
	veorq_u32(veorq_u32((x), (y)), (z))

#define I_SIMD(x, y, z) \
	veorq_u32((y), vorrq_u32((x), vmvnq_u32((z))))

// NEON SIMD版本的循环左移
#define ROTATELEFT_SIMD(num, n) \
	vorrq_u32(vshlq_n_u32((num), (n)), vshrq_n_u32((num), 32 - (n)))

// NEON SIMD版本的MD5四轮轮函数
#define FF_SIMD(a, b, c, d, x, s, ac) { \
	(a) = vaddq_u32((a), F_SIMD((b), (c), (d))); \
	(a) = vaddq_u32((a), (x)); \
	(a) = vaddq_u32((a), vdupq_n_u32(ac)); \
	(a) = ROTATELEFT_SIMD((a), (s)); \
	(a) = vaddq_u32((a), (b)); \
}

#define GG_SIMD(a, b, c, d, x, s, ac) { \
	(a) = vaddq_u32((a), G_SIMD((b), (c), (d))); \
	(a) = vaddq_u32((a), (x)); \
	(a) = vaddq_u32((a), vdupq_n_u32(ac)); \
	(a) = ROTATELEFT_SIMD((a), (s)); \
	(a) = vaddq_u32((a), (b)); \
}

#define HH_SIMD(a, b, c, d, x, s, ac) { \
	(a) = vaddq_u32((a), H_SIMD((b), (c), (d))); \
	(a) = vaddq_u32((a), (x)); \
	(a) = vaddq_u32((a), vdupq_n_u32(ac)); \
	(a) = ROTATELEFT_SIMD((a), (s)); \
	(a) = vaddq_u32((a), (b)); \
}

#define II_SIMD(a, b, c, d, x, s, ac) { \
	(a) = vaddq_u32((a), I_SIMD((b), (c), (d))); \
	(a) = vaddq_u32((a), (x)); \
	(a) = vaddq_u32((a), vdupq_n_u32(ac)); \
	(a) = ROTATELEFT_SIMD((a), (s)); \
	(a) = vaddq_u32((a), (b)); \
}

/**
 * MD5Hash_SIMD4: 将4个输入字符串同时转换成MD5
 * @param inputs 输入字符串数组
 * @param[out] states 用于给调用者传递额外的返回值，即4组MD5结果
 */
void MD5Hash_SIMD4(const string inputs[4], bit32 states[4][4])
{
	Byte *paddedMessages[4];
	int messageLengths[4];

	paddedMessages[0] = StringProcess(inputs[0], &messageLengths[0]);
	paddedMessages[1] = StringProcess(inputs[1], &messageLengths[1]);
	paddedMessages[2] = StringProcess(inputs[2], &messageLengths[2]);
	paddedMessages[3] = StringProcess(inputs[3], &messageLengths[3]);

	// 若4个消息填充后的长度不同，则回退到串行版本，保证正确性
	// 这里直接复用已经预处理好的 paddedMessage，避免重复执行 StringProcess
	if (messageLengths[1] != messageLengths[0] ||
		messageLengths[2] != messageLengths[0] ||
		messageLengths[3] != messageLengths[0])
	{
		MD5HashFromPadded(paddedMessages[0], messageLengths[0], states[0]);
		MD5HashFromPadded(paddedMessages[1], messageLengths[1], states[1]);
		MD5HashFromPadded(paddedMessages[2], messageLengths[2], states[2]);
		MD5HashFromPadded(paddedMessages[3], messageLengths[3], states[3]);

		delete[] paddedMessages[0];
		delete[] paddedMessages[1];
		delete[] paddedMessages[2];
		delete[] paddedMessages[3];
		return;
	}

	int n_blocks = messageLengths[0] / 64;

	uint32x4_t state0 = vdupq_n_u32(0x67452301);
	uint32x4_t state1 = vdupq_n_u32(0xefcdab89);
	uint32x4_t state2 = vdupq_n_u32(0x98badcfe);
	uint32x4_t state3 = vdupq_n_u32(0x10325476);

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		uint32x4_t x[16];

		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			int offset = 4 * i1 + i * 64;

			bit32 word0 = LoadLE32(paddedMessages[0] + offset);
			bit32 word1 = LoadLE32(paddedMessages[1] + offset);
			bit32 word2 = LoadLE32(paddedMessages[2] + offset);
			bit32 word3 = LoadLE32(paddedMessages[3] + offset);

			uint32x4_t v = vdupq_n_u32(0);
			v = vsetq_lane_u32(word0, v, 0);
			v = vsetq_lane_u32(word1, v, 1);
			v = vsetq_lane_u32(word2, v, 2);
			v = vsetq_lane_u32(word3, v, 3);
			x[i1] = v;
		}

		uint32x4_t a = state0, b = state1, c = state2, d = state3;

		/* Round 1 */
		FF_SIMD(a, b, c, d, x[0], s11, 0xd76aa478);
		FF_SIMD(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF_SIMD(c, d, a, b, x[2], s13, 0x242070db);
		FF_SIMD(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF_SIMD(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF_SIMD(d, a, b, c, x[5], s12, 0x4787c62a);
		FF_SIMD(c, d, a, b, x[6], s13, 0xa8304613);
		FF_SIMD(b, c, d, a, x[7], s14, 0xfd469501);
		FF_SIMD(a, b, c, d, x[8], s11, 0x698098d8);
		FF_SIMD(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF_SIMD(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF_SIMD(b, c, d, a, x[11], s14, 0x895cd7be);
		FF_SIMD(a, b, c, d, x[12], s11, 0x6b901122);
		FF_SIMD(d, a, b, c, x[13], s12, 0xfd987193);
		FF_SIMD(c, d, a, b, x[14], s13, 0xa679438e);
		FF_SIMD(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG_SIMD(a, b, c, d, x[1], s21, 0xf61e2562);
		GG_SIMD(d, a, b, c, x[6], s22, 0xc040b340);
		GG_SIMD(c, d, a, b, x[11], s23, 0x265e5a51);
		GG_SIMD(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG_SIMD(a, b, c, d, x[5], s21, 0xd62f105d);
		GG_SIMD(d, a, b, c, x[10], s22, 0x2441453);
		GG_SIMD(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG_SIMD(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG_SIMD(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG_SIMD(d, a, b, c, x[14], s22, 0xc33707d6);
		GG_SIMD(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG_SIMD(b, c, d, a, x[8], s24, 0x455a14ed);
		GG_SIMD(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG_SIMD(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG_SIMD(c, d, a, b, x[7], s23, 0x676f02d9);
		GG_SIMD(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH_SIMD(a, b, c, d, x[5], s31, 0xfffa3942);
		HH_SIMD(d, a, b, c, x[8], s32, 0x8771f681);
		HH_SIMD(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH_SIMD(b, c, d, a, x[14], s34, 0xfde5380c);
		HH_SIMD(a, b, c, d, x[1], s31, 0xa4beea44);
		HH_SIMD(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH_SIMD(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH_SIMD(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH_SIMD(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH_SIMD(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH_SIMD(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH_SIMD(b, c, d, a, x[6], s34, 0x4881d05);
		HH_SIMD(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH_SIMD(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH_SIMD(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH_SIMD(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II_SIMD(a, b, c, d, x[0], s41, 0xf4292244);
		II_SIMD(d, a, b, c, x[7], s42, 0x432aff97);
		II_SIMD(c, d, a, b, x[14], s43, 0xab9423a7);
		II_SIMD(b, c, d, a, x[5], s44, 0xfc93a039);
		II_SIMD(a, b, c, d, x[12], s41, 0x655b59c3);
		II_SIMD(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II_SIMD(c, d, a, b, x[10], s43, 0xffeff47d);
		II_SIMD(b, c, d, a, x[1], s44, 0x85845dd1);
		II_SIMD(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II_SIMD(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II_SIMD(c, d, a, b, x[6], s43, 0xa3014314);
		II_SIMD(b, c, d, a, x[13], s44, 0x4e0811a1);
		II_SIMD(a, b, c, d, x[4], s41, 0xf7537e82);
		II_SIMD(d, a, b, c, x[11], s42, 0xbd3af235);
		II_SIMD(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II_SIMD(b, c, d, a, x[9], s44, 0xeb86d391);

		state0 = vaddq_u32(state0, a);
		state1 = vaddq_u32(state1, b);
		state2 = vaddq_u32(state2, c);
		state3 = vaddq_u32(state3, d);
	}

	bit32 state0_array[4], state1_array[4], state2_array[4], state3_array[4];
	vst1q_u32(state0_array, state0);
	vst1q_u32(state1_array, state1);
	vst1q_u32(state2_array, state2);
	vst1q_u32(state3_array, state3);

	// 下面的处理，在理解上较为复杂
	states[0][0] = ByteSwap32(state0_array[0]);
	states[0][1] = ByteSwap32(state1_array[0]);
	states[0][2] = ByteSwap32(state2_array[0]);
	states[0][3] = ByteSwap32(state3_array[0]);

	states[1][0] = ByteSwap32(state0_array[1]);
	states[1][1] = ByteSwap32(state1_array[1]);
	states[1][2] = ByteSwap32(state2_array[1]);
	states[1][3] = ByteSwap32(state3_array[1]);

	states[2][0] = ByteSwap32(state0_array[2]);
	states[2][1] = ByteSwap32(state1_array[2]);
	states[2][2] = ByteSwap32(state2_array[2]);
	states[2][3] = ByteSwap32(state3_array[2]);

	states[3][0] = ByteSwap32(state0_array[3]);
	states[3][1] = ByteSwap32(state1_array[3]);
	states[3][2] = ByteSwap32(state2_array[3]);
	states[3][3] = ByteSwap32(state3_array[3]);

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessages[0];
	delete[] paddedMessages[1];
	delete[] paddedMessages[2];
	delete[] paddedMessages[3];
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(const string &input, bit32 *state)
{
	Byte *paddedMessage;
	int messageLength;

	paddedMessage = StringProcess(input, &messageLength);
	MD5HashFromPadded(paddedMessage, messageLength, state);

	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
}