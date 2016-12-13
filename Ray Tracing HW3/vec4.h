#pragma once

#include <math.h>

#include "vec2.h"
#include "vec3.h"
#include "vec4.h"

class vec4
{
public:
	union {
		struct {
			float x;
			float y;
			float z;
			float w;
		};

		float v[4];
	};

	vec4() :x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}

	vec4(float f) :x(f), y(f), z(f), w(f) {}

	vec4(vec3 f3) :x(f3.x), y(f3.y), z(f3.z), w(1.0f) {}

	vec4(float x, float y, float z, float w) :x(x), y(y), z(z), w(w) {}

	vec4& operator+=(const vec4& o)
	{
		x += o.x;
		y += o.y;
		z += o.z;
		w += o.w;
		return *this;
	}

	vec4& operator-=(const vec4& o)
	{
		x -= o.x;
		y -= o.y;
		z -= o.z;
		w -= o.w;
		return *this;
	}

	vec4& operator*=(const vec4& o)
	{
		x *= o.x;
		y *= o.y;
		z *= o.z;
		w *= o.w;
		return *this;
	}

	vec4& operator/=(const vec4& o)
	{
		x /= o.x;
		y /= o.y;
		z /= o.z;
		w /= o.w;
		return *this;
	}

	vec4& operator%=(const vec4& o)
	{
		x = fmodf(x, o.x);
		y = fmodf(y, o.y);
		z = fmodf(z, o.z);
		w = fmodf(w, o.w);
		return *this;
	}

	vec4 operator+(const vec4& o) const
	{
		return vec4(x + o.x, y + o.y, z + o.z, w + o.w);
	}

	vec4 operator-(const vec4& o) const
	{
		return vec4(x - o.x, y - o.y, z - o.z, w - o.w);
	}

	vec4 operator*(const vec4& o) const
	{
		return vec4(x * o.x, y * o.y, z * o.z, w * o.w);
	}

	vec4 operator/(const vec4& o) const
	{
		return vec4(x / o.x, y / o.y, z / o.z, w / o.w);
	}

	vec4 operator%(const vec4& o) const
	{
		return vec4(fmodf(x, o.x), fmodf(y, o.y), fmodf(z, o.z), fmodf(w, o.w));
	}


	vec4 operator+() const
	{
		return vec4(+x, +y, +z, +w);
	}

	vec4 operator-() const
	{
		return vec4(-x, -y, -z, -w);
	}

	vec4 operator!() const
	{
		return vec4(-x, -y, -z, +w);
	}

	float distance(const vec4& o) const
	{
		return (*this - o).norm();
	}

	float dot(const vec4& o) const
	{
		return x * o.x + y * o.y + z * o.z + w * o.w;
	}

	float norm() const
	{
		return sqrtf(this->dot(*this));
	}

	float norm2() const
	{
		return this->dot(*this);
	}

	vec4 normalize() const
	{
		return *this / norm();
	}

};
