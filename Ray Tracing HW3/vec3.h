#pragma once

#include <math.h>
#include <stdlib.h>

class vec3
{
public:
	float x;
	float y;
	float z;

	vec3()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	static vec3 random()
	{
		return vec3(
			((float)rand() / RAND_MAX),
			((float)rand() / RAND_MAX),
			((float)rand() / RAND_MAX));
	}

	vec3(float x, float y, float z) :x(x), y(y), z(z) {}

	vec3 operator-() const
	{
		return vec3(-x, -y, -z);
	}

	vec3 operator+(const vec3& operand) const
	{
		return vec3(x + operand.x, y + operand.y, z + operand.z);
	}

	vec3 operator-(const vec3& operand) const
	{
		return vec3(x - operand.x, y - operand.y, z - operand.z);
	}

	vec3 operator*(const vec3& operand) const
	{
		return vec3(x * operand.x, y * operand.y, z * operand.z);
	}

	vec3 operator*(float operand) const
	{
		return vec3(x * operand, y * operand, z * operand);
	}

	void operator-=(const vec3& a)
	{
		x -= a.x;
		y -= a.y;
		z -= a.z;
	}

	void operator+=(const vec3& a)
	{
		x += a.x;
		y += a.y;
		z += a.z;
	}

	void operator*=(const vec3& a)
	{
		x *= a.x;
		y *= a.y;
		z *= a.z;
	}

	void operator*=(float a)
	{
		x *= a;
		y *= a;
		z *= a;
	}

	vec3 operator/(float a) {
		vec3 newVec(0, 0, 0);
		newVec.x = x / a;
		newVec.y = y / a;
		newVec.z = z / a;

		return newVec;
	}

	vec3 operator/(vec3 v) {
		vec3 newVec(0, 0, 0);
		newVec.x = this->x / v.x;
		newVec.y = this->y / v.y;
		newVec.z = this->z / v.z;
		return newVec;
	}

	float length() {
		return sqrt(x*x + y*y + z*z);
	}

	float norm() const
	{
		return sqrtf(x*x + y*y + z*z);
	}

	float norm2() const
	{
		return x*x + y*y + z*z;
	}

	vec3 normalize()
	{
		float oneOverLength = 1.0f / norm();
		x *= oneOverLength;
		y *= oneOverLength;
		z *= oneOverLength;
		return *this;
	}

	vec3 cross(const vec3& operand) const
	{
		return vec3(
			y * operand.z - z * operand.y,
			z * operand.x - x * operand.z,
			x * operand.y - y * operand.x);
	}

	float dot(const vec3& operand) const
	{
		return x * operand.x + y * operand.y + z * operand.z;
	}
};