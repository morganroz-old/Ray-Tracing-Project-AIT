#pragma once

#include <math.h>
#include <stdlib.h>

class vec2
{
public:
	float x;
	float y;

	vec2()
	{
		x = 0.0f;
		y = 0.0f;
	}


	vec2(float x, float y) :x(x), y(y) {}

	vec2 operator-() const
	{
		return vec2(-x, -y);
	}


	vec2 operator+(const vec2& operand) const
	{
		return vec2(x + operand.x, y + operand.y);
	}

	vec2 operator-(const vec2& operand) const
	{
		return vec2(x - operand.x, y - operand.y);
	}

	vec2 operator*(const vec2& operand) const
	{
		return vec2(x * operand.x, y * operand.y);
	}

	vec2 operator*(float operand) const
	{
		return vec2(x * operand, y * operand);
	}

	void operator-=(const vec2& a)
	{
		x -= a.x;
		y -= a.y;
	}

	void operator+=(const vec2& a)
	{
		x += a.x;
		y += a.y;
	}

	void operator*=(const vec2& a)
	{
		x *= a.x;
		y *= a.y;
	}

	void operator*=(float a)
	{
		x *= a;
		y *= a;
	}

	float norm()
	{
		return sqrtf(x*x + y*y);
	}

	float norm2()
	{
		return x*x + y*y;
	}

	vec2 normalize()
	{
		float oneOverLength = 1.0f / norm();
		x *= oneOverLength;
		y *= oneOverLength;
		return *this;
	}

	static vec2 random()
	{
		return vec2(
			((float)rand() / RAND_MAX) * 2 - 1,
			((float)rand() / RAND_MAX) * 2 - 1);
	}
};