/* Lee sang eun */

#include <iostream>
#include <fstream>
#include <gl/freeglut.h>

#include "cuda_header.cuh"
#include "DefConstants.cuh"


using ubyte = unsigned char;


ubyte hstVolume[VOLUME_SIZE] = { 0 };
ubyte tex[TEXTURE_SIZE] = { 0 };


// Camera basis
float eye[3] = { 0.f };
float at[3] = { 128.f, 128.f, 112.f };
float up[3] = { 0.f, 0.f, -1.f };

float dir[3] = { 0.f };
float cross[3] = { 0.f };
float u[3] = { 0.f };

__constant__ float devEye[3];
__constant__ float devDir[3];
__constant__ float devCross[3];
__constant__ float devU[3];


// Transfer function
float hstColorTF[256 * 4];
__constant__ float devColorTF[256 * 4];



////////////////////////////////////////////////////////////
// device functions
////////////////////////////////////////////////////////////

// get offset
__device__ __host__
int getOffset1D(const int px, const int py, const int pz)
{
	return (pz * VOLUME_HEIGHT * VOLUME_WIDTH + py * VOLUME_WIDTH + px);
}
__device__ __host__
int getOffset1D(const int p[3])
{
	return (p[2] * VOLUME_HEIGHT * VOLUME_WIDTH + p[1] * VOLUME_WIDTH + p[0]);
}


// interpolation
__device__
const float nearest_interpolate(const float point[3], const ubyte* volume)
{
	int p[3] = { (int)point[0], (int)point[1], (int)point[2] };
	int offset = getOffset1D(p);
	return volume[offset];
}

__device__
const float linear_interpolate(const float point[3], const ubyte* volume)
{
	if ((0 < point[0] && point[0] < (VOLUME_WIDTH - 1)) &&
		(0 < point[1] && point[1] < (VOLUME_HEIGHT - 1)) &&
		(0 < point[2] && point[2] < (VOLUME_DEPTH - 1))) {

		float dx = point[0] - (int)(point[0]);
		float dy = point[1] - (int)(point[1]);
		float dz = point[2] - (int)(point[2]);

		int smallX = (int)floorf(point[0]);
		int smallY = (int)floorf(point[1]);
		int smallZ = (int)floorf(point[2]);
		int bigX = (int)ceilf(point[0]);
		int bigY = (int)ceilf(point[1]);
		int bigZ = (int)ceilf(point[2]);

		int offset_sxsysz = getOffset1D(smallX, smallY, smallZ);
		int offset_bxsysz = getOffset1D(bigX, smallY, smallZ);
		int offset_sxbysz = getOffset1D(smallX, bigY, smallZ);
		int offset_bxbysz = getOffset1D(bigX, bigY, smallZ);

		int offset_sxsybz = getOffset1D(smallX, smallY, bigZ);
		int offset_bxsybz = getOffset1D(bigX, smallY, bigZ);
		int offset_sxbybz = getOffset1D(smallX, bigY, bigZ);
		int offset_bxbybz = getOffset1D(bigX, bigY, bigZ);

		float invUbyte = 1.0f / 255.f;
		float result =
			(1 - dz) * (1 - dy) * (1 - dx) * volume[offset_sxsysz] +
			(1 - dz) * (1 - dy) * dx * volume[offset_bxsysz] +
			(1 - dz) * dy * (1 - dx) * volume[offset_sxbysz] +
			(1 - dz) * dy * dx * volume[offset_bxbysz] +
			dz * (1 - dy) * (1 - dx) * volume[offset_sxsybz] +
			dz * (1 - dy) * dx * volume[offset_bxsybz] +
			dz * dy * (1 - dx) * volume[offset_sxbybz] +
			dz * dy * dx * volume[offset_bxbybz];

		return (result * invUbyte);
	}
	else
		return 0;
}


// vector operator
__device__ __host__
float vectorLength(const float v[3])
{
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

__device__ __host__
void vectorNormalize(float v[3])
{
	float len = vectorLength(v);
	if (len == 0)
		return;
	float invLen = 1.f / len;
	v[0] *= invLen;
	v[1] *= invLen;
	v[2] *= invLen;
}

__device__ __host__
float dotProduct(const float a[3], const float b[3])
{
	return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

__device__ __host__
void crossProduct(const float a[3], const float b[3], float res[3])
{
	res[0] = a[1] * b[2] - a[2] * b[1];
	res[1] = a[2] * b[0] - a[0] * b[2];
	res[2] = a[0] * b[1] - a[1] * b[0];
}


// truncate number
__device__
int truncate(const float n) {
	int rst = (int)__max(0.f, __min(n, 255.f));
	return rst;
}


// ray-box intersection
__device__
void rayBoxIntersect(const float rayStart[3], float minMax[2])
{
	float maxIdx[3], minIdx[3];

	//초기화 해주고
	maxIdx[0] = maxIdx[1] = maxIdx[2] = -100000;
	minIdx[0] = minIdx[1] = minIdx[2] = 100000;

	float tMax, tMin;
	if (devDir[0] == 0.f) {
		maxIdx[0] = 255.f;
		minIdx[0] = 0.f;
	}
	else {
		tMax = (255.f - rayStart[0]) / devDir[0];
		tMin = (0.f - rayStart[0]) / devDir[0];
		maxIdx[0] = fmaxf(tMax, tMin); // 각축의 max 
		minIdx[0] = fminf(tMax, tMin); // 각축의 min
	}

	if (devDir[1] == 0.f) {
		maxIdx[1] = 255.f;
		minIdx[1] = 0.f;
	}
	else {
		tMax = (255.f - rayStart[1]) / devDir[1];
		tMin = (0.f - rayStart[1]) / devDir[1];
		maxIdx[1] = fmaxf(tMax, tMin); // 각축의 max 
		minIdx[1] = fminf(tMax, tMin); // 각축의 min
	}

	if (devDir[2] == 0.f) {
		maxIdx[2] = 224.f;
		minIdx[2] = 0.f;
	}
	else {
		tMax = (224.f - rayStart[2]) / devDir[2];
		tMin = (0.f - rayStart[2]) / devDir[2];
		maxIdx[2] = fmaxf(tMax, tMin); // 각축의 max 
		minIdx[2] = fminf(tMax, tMin); // 각축의 min
	}

	minMax[0] = fmaxf(fmaxf(minIdx[0], minIdx[1]), minIdx[2]);
	minMax[1] = fminf(fminf(maxIdx[0], maxIdx[1]), maxIdx[2]);
}



////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////

// MIP kernel
__global__
void kernelMIP(ubyte* devVolume, float* devImg) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	float rayStart[3];
	rayStart[0] = devEye[0] + devCross[0] * (x - TEXTURE_WIDTH * 0.5f) + devU[0] * (y - TEXTURE_HEIGHT * 0.5f);
	rayStart[1] = devEye[1] + devCross[1] * (x - TEXTURE_WIDTH * 0.5f) + devU[1] * (y - TEXTURE_HEIGHT * 0.5f);
	rayStart[2] = devEye[2] + devCross[2] * (x - TEXTURE_WIDTH * 0.5f) + devU[2] * (y - TEXTURE_HEIGHT * 0.5f);
	printf("%f %f %f\n", rayStart[0], rayStart[1], rayStart[2]);

	float max = 0;
	for (float t = 0.f; t < 300.f; t += 1.f)
	{
		float p[3];
		p[0] = rayStart[0] + devDir[0] * t;
		p[1] = rayStart[1] + devDir[1] * t;
		p[2] = rayStart[2] + devDir[2] * t;

		int px = (int)p[0];
		int py = (int)p[1];
		int pz = (int)p[2];

		if ((px > 254 || px < 0) || (py > 254 || py < 0) || (pz > 223 || pz < 0))
			continue;

		float vol = linear_interpolate(p, devVolume);
		max = __max(vol, max);
	}

	int devImgOffset = (y * 256 + x) * 3;
	devImg[devImgOffset] = max;
	devImg[devImgOffset + 1] = max;
	devImg[devImgOffset + 2] = max;
}

// DVR kernel
__global__
void kernelVR(ubyte* devVolume, float* devImg) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	float rayStart[3];
	rayStart[0] = devEye[0] + devCross[0] * (x - TEXTURE_WIDTH * 0.5f) + devU[0] * (y - TEXTURE_HEIGHT * 0.5f);
	rayStart[1] = devEye[1] + devCross[1] * (x - TEXTURE_WIDTH * 0.5f) + devU[1] * (y - TEXTURE_HEIGHT * 0.5f);
	rayStart[2] = devEye[2] + devCross[2] * (x - TEXTURE_WIDTH * 0.5f) + devU[2] * (y - TEXTURE_HEIGHT * 0.5f);

	// min = 0, max = 1
	float minMax[2] = { 0.f };
	rayBoxIntersect(rayStart, minMax);
	
	float totalColor[3] = { 0.f, 0.f, 0.f };
	float totalAlpha = 0.f;
	
	for (float k = minMax[0]; k < minMax[1]; k += STEP) {
		float ray[3];
		ray[0] = rayStart[0] + devDir[0] * k;
		ray[1] = rayStart[1] + devDir[1] * k;
		ray[2] = rayStart[2] + devDir[2] * k;

		int px = (int)ray[0];
		int py = (int)ray[1];
		int pz = (int)ray[2];

		if ((px > 254 || px < 0) || (py > 254 || py < 0) || (pz > 223 || pz < 0))
			continue;

		float density = linear_interpolate(ray, devVolume);
		const int den = truncate(density * 255); // truncate
		const float color[3] = { devColorTF[den * 4 + 0], devColorTF[den * 4 + 1], devColorTF[den * 4 + 2] };
		const float alpha = devColorTF[den * 4 + 3];

		totalColor[0] += color[0] * alpha * (1.f - totalAlpha);
		totalColor[1] += color[1] * alpha * (1.f - totalAlpha);
		totalColor[2] += color[2] * alpha * (1.f - totalAlpha);
		totalAlpha += alpha * (1.f - totalAlpha);

		// ealry ray termination
		if (totalAlpha > 0.99f)
			break;
	}

	int devImgOffset = (y * 256 + x) * 3;
	devImg[devImgOffset] = totalColor[0];
	devImg[devImgOffset + 1] = totalColor[1];
	devImg[devImgOffset + 2] = totalColor[2];
}



////////////////////////////////////////////////////////////
// host functions
////////////////////////////////////////////////////////////

// set basis vector
void setBasis(float eye[3], float at[3], float up[3], float dir[3], float cross[3], float u[3])
{
	printf("input eye : ");
	scanf("%f %f %f", &eye[0], &eye[1], &eye[2]);

	dir[0] = at[0] - eye[0];
	dir[1] = at[1] - eye[1];
	dir[2] = at[2] - eye[2];
	vectorNormalize(dir);

	crossProduct(up, dir, cross);
	vectorNormalize(cross);

	crossProduct(dir, cross, u);
	vectorNormalize(u);
	printf("%f %f %f\n", u[0], u[1], u[2]);

	size_t vectorSize = sizeof(float) * 3;
	cudaMemcpyToSymbol(devEye, eye, vectorSize);
	cudaMemcpyToSymbol(devDir, dir, vectorSize);
	cudaMemcpyToSymbol(devCross, cross, vectorSize);
	cudaMemcpyToSymbol(devU, u, vectorSize);
}

// set transfer function
void setTransferFunction(const int a1, const int a2)
{
	// color
	for (int i = 0; i < a1; ++i) {
		hstColorTF[i * 4 + 0] = 0.f;
		hstColorTF[i * 4 + 1] = 0.f;
		hstColorTF[i * 4 + 2] = 0.f;
	}
	for (int i = a1; i < a2; ++i) {
		hstColorTF[i * 4 + 0] = (i - a1 + 1.f) * (1.f / (a2 - a1));
		hstColorTF[i * 4 + 1] = (i - a1 + 1.f) * (1.f / (a2 - a1));
		hstColorTF[i * 4 + 2] = (i - a1 + 1.f) * (1.f / (a2 - a1));
	}
	for (int i = a2; i < 256; ++i) {
		hstColorTF[i * 4 + 0] = 1.f;
		hstColorTF[i * 4 + 1] = 1.f;
		hstColorTF[i * 4 + 2] = 1.f;
	}

	// alpha
	for (int i = 0; i < a1; ++i)
		hstColorTF[i * 4 + 3] = 0.f;
	for (int i = a1; i < a2; ++i)
		hstColorTF[i * 4 + 3] = (i - a1 + 1.f) * (1.f / (a2 - a1));
	for (int i = a2; i < 256; ++i)
		hstColorTF[i * 4 + 3] = 1.f;

	// alphaCorrection
	for (int i = 0; i < a1; ++i)
		hstColorTF[i * 4 + 3] = 0.f;
	for (int i = a1; i < a2; ++i)
		hstColorTF[i * 4 + 3] = 1.f - powf(1.f - hstColorTF[i * 4 + 3], STEP);;
	for (int i = a2; i < 256; ++i)
		hstColorTF[i * 4 + 3] = 1.f;

	cudaMemcpyToSymbol(devColorTF, hstColorTF, sizeof(float) * 256 * 4);
}

// gl display
void displayFunc()
{
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_RGB,
		GL_FLOAT, tex);

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);

	glBegin(GL_QUADS);

	glTexCoord2f(0.f, 0.f); glVertex3f(-0.8f, -0.8f, 0.f);
	glTexCoord2f(1.f, 0.f); glVertex3f(0.8f, -0.8f, 0.f);
	glTexCoord2f(1.f, 1.f); glVertex3f(0.8f, 0.8f, 0.f);
	glTexCoord2f(0.f, 1.f); glVertex3f(-0.8f, 0.8f, 0.f);

	glEnd();

	glutSwapBuffers();
}



////////////////////////////////////////////////////////////
// main function
////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// set volume
	std::ifstream input("./volume/Bighead.den", std::ios::binary | std::ios::in);
	if (!input)
		return 0;
	input.read(reinterpret_cast<char *>(hstVolume), VOLUME_SIZE);
	input.close();

	// set volume on device
	ubyte* devVolume = nullptr;
	cudaMalloc(&devVolume, VOLUME_SIZE * sizeof(ubyte));
	cudaMemcpy(devVolume, hstVolume, VOLUME_SIZE * sizeof(ubyte), cudaMemcpyKind::cudaMemcpyHostToDevice);

	// set texture on device
	float* devImg = nullptr;
	cudaMalloc(&devImg, TEXTURE_SIZE * sizeof(float));

	// set camera basis
	setBasis(eye, at, up, dir, cross, u);

	printf("dir %f %f %f\n", dir[0], dir[1], dir[2]);
	printf("cross %f %f %f\n", cross[0], cross[1], cross[2]);
	printf("u %f %f %f\n", u[0], u[1], u[2]);

	// set transfer function
	setTransferFunction(100, 140);

	// call kernel
	dim3 gridDim{ 16, 16, 1 };
	dim3 blockDim{ 16, 16, 1 };
	//kernelMIP << <gridDim, blockDim >> > (devVolume, devImg);
	kernelVR << <gridDim, blockDim >> > (devVolume, devImg);

	cudaDeviceSynchronize();
	cudaMemcpy(tex, devImg, TEXTURE_SIZE * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// free device pointer variable
	cudaFree(devImg);
	cudaFree(devVolume);

	// set glut
	glutInit(&argc, argv);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow("Volume Rendering");

	glutDisplayFunc(displayFunc);
	glutMainLoop();

	return 0;
}