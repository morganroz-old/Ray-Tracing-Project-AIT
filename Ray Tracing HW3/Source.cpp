
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		 
#include <GL/freeglut.h>	
#endif

#include<cassert>

#include <vector>
#include "vec2.h"
#include "vec3.h"
#include "vec4.h"
#include "mat4x4.h"

const unsigned int windowWidth = 512, windowHeight = 512;

int majorVersion = 3, minorVersion = 0;


// image to be computed by ray tracing
// memory buffer where we compute the result of ray tracing
vec3 image[windowWidth * windowHeight];


// Light Heirarchy
class LightSource {
public:
	virtual vec3 getPowerDensityAt(vec3 x) = 0;
	virtual vec3 getLightDirAt(vec3 x) = 0;
	virtual float getDistanceFrom(vec3 x) = 0;
};

class DirectionalLight : public LightSource {
private:
	vec3 direction;
	vec3 powerDensity;

public:
	DirectionalLight(vec3 powerDensity, vec3 direction) : direction(direction), powerDensity(powerDensity) {}

	vec3 getPowerDensityAt(vec3 x) {
		return powerDensity;
	}

	vec3 getLightDirAt(vec3 x) {
		return direction;
	}

	float getDistanceFrom(vec3 x) {
		return FLT_MAX;
	}
};

class PointLight : public LightSource {
private:
	vec3 position;
	vec3 powerDensity;

public:
	PointLight(vec3 powerDensity, vec3 position) : position(position), powerDensity(powerDensity) {}

	vec3 getPowerDensityAt(vec3 x) {
		float distance = getDistanceFrom(x);
		return powerDensity / (distance * distance * 4 * M_PI);
	}

	vec3 getLightDirAt(vec3 x) {
		return (x - position).normalize();
	}

	float getDistanceFrom(vec3 x) {
		return (position - x).length();
	}
};


// simple material class, with object color, and headlight shading
class Material
{
protected:
	vec3 frontColor;
	vec3 backColor;


	vec3 reflectance;
	float  refractiveIndex;
	vec3 kd;
	vec3 ks;
	float shininess;

public:
	Material(vec3 fc, vec3 bc) : frontColor(fc), backColor(bc) {}

	boolean isReflective() {
		if (reflectance.x != 0 || reflectance.y != 0 || reflectance.z != 0) {
			return true;
		}
		else return false;
	}

	boolean isRefractive() {
		if (refractiveIndex != 0) {
			return true;
		}
		else return false;
	}

	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		float dotProd = viewDir.dot(normal);
		if (dotProd >= 0)
			return frontColor;
		else
			return backColor;
	}

	vec3 getFrontColor() { return frontColor; }
	vec3 getBackColor() { return backColor; }

	virtual vec3 shade(vec3 position, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 powerDesity) {
		return getColor(position, normal, viewDir);
	}

	virtual vec3 getReflectionDir(vec3 inDir, vec3 outDir) { return vec3(0, 0, 0); }
	virtual vec3 getRefractionDir(vec3 inDir, vec3 outDir) { return vec3(0, 0, 0); }

};

class DiffuseMaterial : public Material {

public:
	DiffuseMaterial(vec3 fc, vec3 bc) : Material(fc, bc) {

	}

	vec3 shade(vec3 position, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 powerDensity) {
		float dotProd = normal.dot(lightDir);
		vec3 materialColor;
		if (dotProd > 0) {
			materialColor = getColor(position, normal, viewDir);
			return powerDensity * (materialColor * dotProd);
		}
		else {
			return vec3(0, 0, 0);
		}

	}

	
};

class SpecularMaterial : public Material {
	float shininess;
public:
	SpecularMaterial(vec3 fc, vec3 bc, float shine) : Material(fc, bc), shininess(shine) {}

	vec3 shade(vec3 position, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 powerDensity) {
		vec3 materialColor;
		if (viewDir.dot(normal) > 0) {
			materialColor = frontColor;
		}
		else {
			materialColor = backColor;
		}
		vec3 halfwayVec = (viewDir + lightDir).normalize();
		float dotProd = halfwayVec.dot(normal);
		if (dotProd > 0) {
			return powerDensity * (materialColor * pow(dotProd, shininess));
		}
		else
			return vec3(0, 0, 0);
	}
};

class Metal : public SpecularMaterial {
	vec3 r0;

public:
	Metal(vec3  refractiveIndex, vec3  extinctionCoefficient) : SpecularMaterial(vec3(0,0,0), vec3(0,0,0), 0) {

		vec3 ones(1, 1, 1);
		vec3 rim1 = refractiveIndex - ones;
		vec3 rip1 = refractiveIndex + ones;
		vec3 k2 = extinctionCoefficient * extinctionCoefficient;
		r0 = (rim1*rim1 + k2) / (rip1*rip1 + k2);
	}

	struct Event {
		vec3 reflectionDir;
		vec3 reflectance;
	};

	Event evaluateEvent(vec3 inDir, vec3 normal) {
		Event e;

		float cosa = -normal.dot(inDir);
		vec3 perp = -normal * cosa;
		vec3 parallel = inDir - perp;
		e.reflectionDir = parallel - perp;

		e.reflectance = (r0 + (vec3(1, 1, 1) - r0) * pow(1 - cosa, 5));

		return e;
	}
};

class Dielectric : public Material {
	float  refractiveIndex;
	float  r0;
public:
	Dielectric(float refractiveIndex, Material* material) : refractiveIndex(refractiveIndex), Material(*material) {
		r0 = (refractiveIndex - 1) * (refractiveIndex - 1) 
			/ (refractiveIndex + 1) / (refractiveIndex + 1);
	}

	struct Event {
		vec3 reflectionDir;
		vec3 refractionDir;
		float reflectance;
		float transmittance;
	};
	
	Event evaluateEvent(vec3 inDir, vec3 normal) {
		Event e;
		// computation on next slide
		float cosa = -normal.dot(inDir);
		vec3 perp = -normal * cosa;
		vec3 parallel = inDir - perp;
		e.reflectionDir = parallel - perp;

		float ri = refractiveIndex;
		if (cosa < 0) { cosa = -cosa; normal = -normal; ri = 1 / ri; }
		float disc = 1 - (1 - cosa * cosa) / ri / ri;
		if (disc < 0)
			e.reflectance = 1;
		else {
			float cosb = sqrt(disc);
			e.refractionDir = parallel / ri - normal * cosb;
			e.reflectance = (r0 + (1 - r0) * pow(1 - cosa, 5));
		}
		e.transmittance = 1 - e.reflectance;

		return e;
	}
};



// Custom Materials
vec3 snoiseGrad(vec3 r) {
	vec3 s = vec3(7502, 22777, 4767); // random seed, kind of
	vec3 f = vec3(0.0, 0.0, 0.0);
	int control = 32768;
	double intPart;
	for (int i = 0; i<16; i++) {
		f += s * cos(s.dot(r) / 65536.0); // add a bunch of random sines
		vec3 modvec = vec3((int)s.x % control, (int)s.y % control, (int)s.z % control);
		vec3 floorvec = vec3(floor(s.x / control), floor(s.y / control), floor(s.z / control));
		s = modvec * 2.0 + floorvec;
		// generate next random
	}
	return f / 65536.0;
}

bool keepHit() {
	float random = rand() % 10;
	if (random < 9)
		return true;
	return false;
}

float snoise(vec3 r) {
	unsigned int x = 0x0625DF73;
	unsigned int y = 0xD1B84B45;
	unsigned int z = 0x152AD8D0;
	float f = 0;
	for (int i = 0; i<32; i++) {
		vec3 s(x / (float)0xffffffff,
			y / (float)0xffffffff,
			z / (float)0xffffffff);
		f += sin(s.dot(r));
		x = x << 1 | x >> 31;
		y = y << 1 | y >> 31;
		z = z << 1 | z >> 31;
	}
	return f / 64.0 + 0.5;
}

class PresentBox : public SpecularMaterial {

public:
	PresentBox(vec3 fc, vec3 bc) : SpecularMaterial(fc, bc, 0) {

	}

	vec3 shade(vec3 position, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 powerDensity) {
		normal = normal + snoiseGrad(position * 40.0) * 0.01;

		float dotProd = normal.dot(lightDir);
		vec3 materialColor;
		materialColor = getColor(position, normal, viewDir);
		return powerDensity * (frontColor * dotProd);

	}
};

class EvergreenTree : public DiffuseMaterial {

public:
	EvergreenTree(vec3 fc, vec3 bc) : DiffuseMaterial(fc, bc) { }

	vec3 shade(vec3 position, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 powerDensity) {
		float dotProd = normal.dot(lightDir);
		vec3 materialColor;
			materialColor = getColor(position, normal, viewDir);
			return powerDensity * (frontColor * dotProd);
	}
};

class Oranges : public DiffuseMaterial {

public:
	Oranges() : DiffuseMaterial(vec3(0.66, 0.33, 0), vec3(0.66, 0.33, 0)) {

	}
	vec3 shade(vec3 position, vec3 normal, vec3 viewDir, vec3 lightDir, vec3 powerDensity) {
		normal = normal  + snoiseGrad(position * 300.0) * 0.05;
		return DiffuseMaterial::shade(position, normal, viewDir, lightDir, powerDensity);

	}

};

class Wood : public DiffuseMaterial {

	float scale;
	float turbulence;
	float period;
	float sharpness;

public:
	Wood() :
		DiffuseMaterial(vec3(1, 1, 1), vec3(1, 1, 1))
	{
		scale = 16;
		turbulence = 500;
		period = 8;
		sharpness = 10;
	}
	virtual vec3 getColor(
		vec3 position,
		vec3 normal,
		vec3 viewDir)
	{
		float w = position.x * period + pow(snoise(position * scale), sharpness)*turbulence + 10000.0;
		w -= int(w);
		return ((vec3(1, 0.3, 0) * w + vec3(0.35, 0.1, 0.05) * (1 - w))) * normal.dot(viewDir);
	}
};

class WoodenFloorGlossy : public Wood {
	vec3 r0;

public:
	WoodenFloorGlossy(vec3  refractiveIndex, vec3  extinctionCoefficient) :
		Wood() {

		vec3 ones(1, 1, 1);
		vec3 rim1 = refractiveIndex - ones;
		vec3 rip1 = refractiveIndex + ones;
		vec3 k2 = extinctionCoefficient * extinctionCoefficient;
		r0 = (rim1*rim1 + k2) / (rip1*rip1 + k2);
	}

	struct Event {
		vec3 reflectionDir;
		vec3 reflectance;
	};

	Event evaluateEvent(vec3 inDir, vec3 normal) {
		Event e;

		float cosa = -normal.dot(inDir);
		vec3 perp = -normal * cosa;
		vec3 parallel = inDir - perp;
		e.reflectionDir = parallel - perp;

		e.reflectance = (r0 + (vec3(1, 1, 1) - r0) * pow(1 - cosa, 5));

		return e;
	}
};

// Camera class.
class Camera
{
	vec3 eye;		//< world space camera position
	vec3 lookAt;	//< center of window in world space
	vec3 right;		//< vector from window center to window right-mid (in world space)
	vec3 up;		//< vector from window center to window top-mid (in world space)

public:
	Camera()
	{
		eye = vec3(0, 0, 2);
		lookAt = vec3(0, 0, 1);
		right = vec3(1, 0, 0);
		up = vec3(0, 1, 0);
	}
	vec3 getEye()
	{
		return eye;
	}
	// compute ray through pixel at normalized device coordinates
	vec3 rayDirFromNdc(float x, float y) {
		return (lookAt - eye
			+ right * x
			+ up    * y
			).normalize();
	}
};

// Ray structure.
class Ray
{
public:
	vec3 origin;
	vec3 dir;
	Ray(vec3 o, vec3 d)
	{
		origin = o;
		dir = d;
	}
};

// Hit record structure. Contains all data that describes a ray-object intersection point.
class Hit
{
public:
	Hit()
	{
		t = -1; // negative means it is behind you, so you don't have to do anything with it
	}
	float t;				//< Ray paramter at intersection. Negative means no valid intersection.
	vec3 position;			//< Intersection coordinates.
	vec3 normal;			//< Surface normal at intersection.
	Material* material;		//< Material of intersected surface.
};

// Abstract base class.
class Intersectable
{
protected:
	Material* material;
public:
	Intersectable(Material* material) : material(material) {}
	virtual Hit intersect(const Ray& ray) = 0;
};

// Simple helper class to solve quadratic equations with the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and store the results.
class QuadraticRoots
{
public:
	float t1;
	float t2;
	// Solves the quadratic a*t*t + b*t + c = 0 using the Quadratic Formula [-b +- sqrt(b^2-4ac)] / 2a, and sets members t1 and t2 to store the roots.
	QuadraticRoots(float a, float b, float c)
	{
		float discr = b * b - 4.0 * a * c;
		if (discr < 0) // no roots
		{
			t1 = -1;
			t2 = -1;
			return;
		}
		float sqrt_discr = sqrt(discr);
		t1 = (-b + sqrt_discr) / 2.0 / a;
		t2 = (-b - sqrt_discr) / 2.0 / a;
	}
	// Returns the lesser of the positive solutions, or a negative value if there was no positive solution.
	float getLesserPositive()
	{
		return (0 < t1 && (t2 < 0 || t1 < t2)) ? t1 : t2;
	}
};

class Plane : public Intersectable {
	vec3 point;
	vec3 normal;
public:
	Plane(const vec3& p, const vec3& n, Material* material) : point(p), normal(n), Intersectable(material) {
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		hit.t = (point - ray.origin).dot(normal) / ray.dir.dot(normal);
		
		hit.material = material;
		hit.position = ray.origin + ray.dir * hit.t;
		hit.normal = normal;

		return hit;
	}

	vec3 getPoint() { return point; }

	vec3 getNormal() { return normal; }

	Plane* translate(vec3 move) {
		this->point = move;
		return this;
	}
};

class Quadric : public Intersectable{

	mat4x4 coeffs;

public:
	Quadric(Material* material) : Intersectable(material) {
		coeffs = mat4x4(
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, -1
		); // scaled by reciprocal of value
	}

	bool contains(vec3 r)
	{
		vec4 rhomog = vec4(r);
		float val = rhomog.dot(coeffs * rhomog);
		if (val > 0) return false;
		else return true;
	}

	QuadraticRoots solveQuadratic(const Ray& ray)
	{
		vec4 d = vec4(ray.dir);
		d.w = 0;
		vec4 e = vec4(ray.origin);

		float a = d.dot(coeffs * d);
		float b = (e.dot(coeffs * d) + d.dot(coeffs * e));
		float c = e.dot(coeffs * e);
		return QuadraticRoots(a, b, c);

	}

	vec3 getNormalAt(vec3 r)
	{
		vec4 norm = coeffs * vec4(r) + operator*(vec4(r), coeffs);
		return vec3(norm.x, norm.y, norm.z).normalize();
	}

	Hit intersect(const Ray& ray) {
		float t = solveQuadratic(ray).getLesserPositive(); // we want the closest to the camera

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = getNormalAt(hit.position);

		return hit;
	}

	Quadric* transform(mat4x4 t) {
		coeffs = t.invert() * coeffs * t.invert().transpose();
		return this;
	}

	// Base Shapes
	Quadric* sphere() {
		return this;
	}

	Quadric* cylinder() {
		coeffs._11 = 0;
		return this;
	}

	Quadric* cone() {
		coeffs._11 = -1;
		coeffs._33 = 0;
		return this;
	}

	Quadric* paraboloid() {
		coeffs._11 = 0;
		coeffs._13 = -1;
		return this;
	}

	Quadric* hyperboloid() {
		coeffs._11 = -1;
		return this;
	}

	Quadric* hyperbolicParaboloid() {
		coeffs._11 = 0;
		coeffs._13 = -1;
		coeffs._22 = -1;
		coeffs._33 = 0;
		return this;
	}

	Quadric* hyperbolicCylinder() {
		coeffs._00 = -1;
		coeffs._11 = 0;
		coeffs._33 = 1;
		return this;
	}

	Quadric* parallelPlanes() {
		coeffs._00 = 0;
		coeffs._11 = 1;
		coeffs._22 = 0;
		coeffs._33 = -1;
		return this;
	}

};

class Square : public Intersectable {
	Plane shape;
	std::vector<Quadric*> clippers; // will be two parallel planes to clip the single plane

public:
	Square(Material* material, vec3 normal) : Intersectable(material), shape(Plane(vec3(0, 0, 0), normal, material)) {
		if (normal.x == 0 && normal.y == 0 && (normal.z == 1 || normal.z == -1)) {
			clippers.push_back((new Quadric(material))->parallelPlanes());
			clippers.push_back((new Quadric(material))->parallelPlanes()->transform(
				mat4x4::rotation(shape.getNormal(), M_PI / 2)));
		}
		else if (normal.x == 0 && (normal.y == 1 || normal.y == -1) && normal.z == 0) {
			clippers.push_back((new Quadric(material))->parallelPlanes()->transform(mat4x4::rotation(vec3(1, 0, 0), M_PI/2)));
			clippers.push_back((new Quadric(material))->parallelPlanes()->transform(mat4x4::rotation(vec3(1, 0, 0), M_PI / 2) * 
				mat4x4::rotation(shape.getNormal(), M_PI/2)));
		}
		else if ((normal.x == 1 || normal.x == -1) && normal.y == 0 && normal.z == 0) {
			clippers.push_back((new Quadric(material))->parallelPlanes()->transform(mat4x4::rotation(vec3(0, 1, 0), M_PI / 2)));
			clippers.push_back((new Quadric(material))->parallelPlanes()->transform(mat4x4::rotation(vec3(0, 1, 0), M_PI / 2) *
				mat4x4::rotation(shape.getNormal(), M_PI / 2)));
		}
	}

	Hit intersect(const Ray& ray) {

		Hit hit;

		hit.t = (shape.getPoint() - ray.origin).dot(shape.getNormal()) / ray.dir.dot(shape.getNormal());

		hit.material = material;
		hit.position = ray.origin + ray.dir * hit.t;
		hit.normal = shape.getNormal();

		for (int i = 0; i < clippers.size(); i++) {
			if (!(clippers[i]->contains(hit.position)))
				hit.t = -1;
		}
		return hit;
	}

	Square* transform(mat4x4 t) {
		for (int i = 0; i < clippers.size(); i++) {
			clippers[i]->transform(t);
		}
		return this;
	}

	Square* translate(vec3 move) {
		shape.translate(move);
		for (int i = 0; i < clippers.size(); i++) {
			clippers[i]->transform(mat4x4::translation(move));
		}
		return this;
	}

	Square* rotate(float angle) {
		for (int i = 0; i < clippers.size(); i++) {
			clippers[i]->transform(mat4x4::rotation(shape.getNormal(), angle));
		}
		return this;
	}

	Square* scaling(vec3 scale) {
		for (int i = 0; i < clippers.size(); i++) {
			clippers[i]->transform(mat4x4::scaling(scale));
		}
		return this;
	}
};

class ClippedQuadric : public Intersectable {
	
	Quadric shape;
	Quadric clipper;

public: 

	ClippedQuadric(Material* material) : Intersectable(material), shape(Quadric(material)), clipper(*(Quadric(material).parallelPlanes())) {
	}

	Hit intersect(const Ray& ray) {
		QuadraticRoots roots = shape.solveQuadratic(ray);

		vec3 position1 = ray.origin + ray.dir * roots.t1;
		vec3 position2 = ray.origin + ray.dir * roots.t2;

		if (!clipper.contains(position1)) {
			roots.t1 = -1;
		}
		if (!clipper.contains(position2)) {
			roots.t2 = -1;
		}
			
		// continue with intersection points
		float t = roots.getLesserPositive();

		Hit hit;
		hit.t = t;
		hit.material = material;
		hit.position = ray.origin + ray.dir * t;
		hit.normal = shape.getNormalAt(hit.position);
		return hit;
	}

	ClippedQuadric* transform(mat4x4 t) {
		shape.transform(t);
		clipper.transform(t);
		return this;
	}

	ClippedQuadric* sphere(float height) {
		clipper.transform(mat4x4::scaling(vec3(height, height, height)));
		return this;
	}

	ClippedQuadric* cylinder(float height) {
		shape.cylinder();
		clipper.transform(mat4x4::scaling(vec3(height, height, height)));
		return this;
	}

	ClippedQuadric* cone(float height) {
		shape.cone();
		clipper.transform(mat4x4::scaling(vec3(height, height, height)));
		return this;
	}

	ClippedQuadric* halfCone(float height) {
		shape.cone();
		clipper.transform(mat4x4::scaling(vec3(height/2, height/2, height/2)) * 
			mat4x4::translation(vec3(0, -height/2, 0)));
		return this;
	}

	ClippedQuadric* hyperboloid(float height, bool startInMiddle) {
		shape.hyperboloid();
		if (startInMiddle) {
			clipper.transform(mat4x4::scaling(vec3(height, height, height)) *
				mat4x4::translation(vec3(0, height, 0)));
		}
		else {
			clipper.transform(mat4x4::scaling(vec3(height, height, height)));
		}
		return this;
	}

	ClippedQuadric* paraboloid(float height) {
		shape.paraboloid();
		clipper.transform(mat4x4::scaling(vec3(height, height, height)) * 
			mat4x4::translation(vec3(0,-height,0)));

		return this;
	}

};


class Scene
{
	Camera camera;
	std::vector<Intersectable*> objects;
	std::vector<Material*> materials;
	std::vector<LightSource*> lightSources;

public:

	void CreateChristmasTree() {
		for (int i = 0; i < 5; i++) {
			objects.push_back((new ClippedQuadric(materials[10]))->halfCone(1 - 0.2 * i)->
				transform(mat4x4::scaling(vec3(1.2, 2.2, 1.2)) * mat4x4::translation(vec3(-1.5, 0.15 * i + 1.8, -0.8))));
		}
		// wooden tree trunk
		objects.push_back((new ClippedQuadric(materials[6]))->cylinder(1)->transform(
			mat4x4::scaling(vec3(0.5,0.5,0.5)) * mat4x4::translation(vec3(-1.5,-0.5,-0.8))));

	}

	void createSnowman(vec3 position) {
		// snowman
		objects.push_back((new Quadric(materials[3]))->sphere()->
			transform(mat4x4::scaling(vec3(0.5, 0.5, 0.5)) * mat4x4::translation(position)));
		objects.push_back((new Quadric(materials[3]))->sphere()->
			transform(mat4x4::scaling(vec3(0.4, 0.4, 0.4)) * mat4x4::translation(vec3(0 + position.x, 0.7 + position.y, 0 + position.z))));
		objects.push_back((new Quadric(materials[3]))->sphere()->
			transform(mat4x4::scaling(vec3(0.3, 0.3, 0.3)) * mat4x4::translation(vec3(0 + position.x, 1.25 + position.y, 0 + position.z))));
		// carrot(cone) nose
		objects.push_back((new ClippedQuadric(materials[4]))->halfCone(0.3)->
			transform(mat4x4::scaling(vec3(0.2, 0.6, 0.2)) * mat4x4::rotation(vec3(1, 0, 1), 90) *
				mat4x4::translation(vec3(-.3 + position.x, 1.05 + position.y, 0.5 + position.z))));
		// coal for eyes
		objects.push_back((new Quadric(materials[5]))->sphere()->transform( // right eye
			mat4x4::scaling(vec3(0.05,0.05,0.05)) * 
			mat4x4::translation(vec3(0 + position.x, 1.25 + position.y, 0.27 + position.z))));
		objects.push_back((new Quadric(materials[5]))->sphere()->transform( // left eye
			mat4x4::scaling(vec3(0.05, 0.05, 0.05)) * 
			mat4x4::translation(vec3(-.2 + position.x, 1.25 + position.y, 0.2 + position.z))));
		// wooden stick arms
		objects.push_back((new ClippedQuadric(materials[4]))->cylinder(0.25)->transform( // left arm
			mat4x4::scaling(vec3(0.02, 1, 0.02)) * 
			mat4x4::rotation(vec3(0,0,1), 20) * 
			mat4x4::translation(vec3(-0.6 + position.x, 1.0 + position.y, 0.4 + position.z)))
		);
		objects.push_back((new ClippedQuadric(materials[4]))->cylinder(0.25)->transform( // right arm
			mat4x4::scaling(vec3(0.02, 1, 0.02)) *
			mat4x4::rotation(vec3(0, 0, 1), -20) *
			mat4x4::translation(vec3(0.30 + position.x, 1.0 + position.y, 0.4 + position.z)))
		);

	}

	void createIcicles() {
		for (int i = 0; i < 7; i++) {
			objects.push_back((new Quadric(materials[9]))->paraboloid()->transform(
				mat4x4::scaling(vec3(0.02, 1, 0.02)) * mat4x4::translation(vec3(-0.09 + i*0.03, 1.05, 1.9))));		
		}
	}

	void createOranges(vec3 position = vec3(0,0,0)) {
		objects.push_back((new Quadric(materials[12]))->sphere()->transform(
			mat4x4::scaling(vec3(0.1, 0.1, 0.1)) * 
			mat4x4::translation(vec3(1 + position.x, -0.9 + position.y, 0.5 + position.z))));
		objects.push_back((new Quadric(materials[12]))->sphere()->transform(
			mat4x4::scaling(vec3(0.1, 0.1, 0.1)) *
			mat4x4::translation(vec3(1.2 + position.x, -0.9 + position.y, 0.5 + position.z))));
		objects.push_back((new Quadric(materials[12]))->sphere()->transform(
			mat4x4::scaling(vec3(0.1, 0.1, 0.1)) *
			mat4x4::translation(vec3(1.1 + position.x, -0.75 + position.y, 0.5 + position.z))));
	}

	void decorateTree() {
		// 3 silver baubles
		objects.push_back((new Quadric(materials[13]))->sphere()->transform(
			mat4x4::scaling(vec3(.1, .15, .1)) * mat4x4::translation(vec3(-.98, 1, -0.5))));
		objects.push_back((new Quadric(materials[13]))->sphere()->transform(
			mat4x4::scaling(vec3(.1, .15, .1)) * mat4x4::translation(vec3(-1, 0, 0))));
		objects.push_back((new Quadric(materials[13]))->sphere()->transform(
			mat4x4::scaling(vec3(.1, .15, .1)) * mat4x4::translation(vec3(-1.3, 0.5, -0.05))));	

		// 3 golden bells
		createGoldenBell(vec3(-1.5, 1.1,-0.2));
		createGoldenBell(vec3(-0.7, 0.5, -0.4));
		createGoldenBell(vec3(-1.4, 0, 0.25));
	}

	void createGoldenBell(vec3 position) {
		// a hyperboloid half
		objects.push_back((new ClippedQuadric(materials[8]))->hyperboloid(0.5, true)->transform(
			mat4x4::scaling(vec3(0.1,0.1,0.1)) * mat4x4::rotation(vec3(0, 0, 1), M_PI) * 
			mat4x4::translation(vec3(position.x, position.y, position.z))));
		// paraboloid top
		objects.push_back((new ClippedQuadric(materials[8]))->paraboloid(0.3)->transform(
			mat4x4::scaling(vec3(0.1,0.2,0.1)) * mat4x4::rotation(vec3(0,0,1), M_PI) *
			mat4x4::translation(vec3(0 + position.x,0 + position.y,0 + position.z))));
	}

	void createPresent(vec3 p, vec3 scale) {

		objects.push_back((new Square(materials[14], vec3(0, 0, 1)))->scaling(scale)
			->translate(vec3(0 + p.x, 0 + p.y, 0 + p.z))); // front
		objects.push_back((new Square(materials[14], vec3(0, 0, -1)))->scaling(scale)
			->translate(vec3(0 + p.x, 0 + p.y, -scale.z + p.z))); // back
		objects.push_back((new Square(materials[14], vec3(0, 1, 0)))->scaling(scale)
			->translate(vec3(0 + p.x, scale.y + p.y, -scale.z + p.z))); // top
		objects.push_back((new Square(materials[14], vec3(0, -1, 0)))->scaling(scale)
			->translate(vec3(0 + p.x, -scale.y + p.y, -scale.z + p.z))); // bottom
		objects.push_back((new Square(materials[14], vec3(1, 0, 0)))->scaling(scale)
			->translate(vec3(scale.x + p.x, 0 + p.y, -scale.z + p.z))); // left
		objects.push_back((new Square(materials[14], vec3(-1, 0, 0)))->scaling(scale)
			->translate(vec3(-scale.x + p.x, 0 + p.y, -scale.z + p.z))); // right

	}

	Scene()	
	{
		lightSources.push_back(new DirectionalLight(vec3(1, 1, 1), vec3(-3, -1, -1)));
		lightSources.push_back(new DirectionalLight(vec3(1, 1, 1), vec3(3, -1, 0)));
		lightSources.push_back(new DirectionalLight(vec3(1, 1, 1), vec3(0, -1, 0)));

		// 0  Specular Red Material
		materials.push_back(new SpecularMaterial(vec3(5, 0, 0), vec3(5, 0, 0), 8));
		// 1, 2  Dummy Diffuse Black Materials for use later
		materials.push_back(new DiffuseMaterial(vec3(0,0,0), vec3(0,0,0)));
		materials.push_back(new DiffuseMaterial(vec3(0,0,0), vec3(0,0,0)));
		// 3  snowman material
		materials.push_back(new DiffuseMaterial(vec3(1,1, 1), vec3(1, 1, 1)));
		// 4  carrot nose
		materials.push_back(new DiffuseMaterial(vec3(0.3, 0.05, 0.02), vec3(0.3, 0, 0)));
		// 5  coal eye balls
		materials.push_back(new DiffuseMaterial(vec3(0, 0, 0), vec3(0, 0, 0))); 
		// 6  Diffuse Wooden surface
		materials.push_back(new Wood());
		// 7  Dummy Diffuse Black surface for use later
		materials.push_back(new DiffuseMaterial(vec3(0,0,0), vec3(0,0,0)));
		// 8  Gold Metal
		materials.push_back(new Metal(vec3(0.21, 0.485, 1.29), vec3(3.13, 2.23, 1.76))); 
		// 9  Icicles
		materials.push_back(new Dielectric(1.2, materials[0]));
		// 10  Evergreen
		materials.push_back(new EvergreenTree(vec3(0.1, 0.2, 0.1), vec3(0.1, 0.2, 0.1))); 
		// 11  Reflective Diffuse wooden floor
		materials.push_back(new WoodenFloorGlossy(vec3(0.2, 0.02, 0.02), vec3(0.2,0.2,0.2))); 
		// 12  Oranges - procedural texturing
		materials.push_back(new Oranges()); 
		// 13  Silver Metal
		materials.push_back(new Metal(vec3(0.15, 0.14, 0.13), vec3(3.7, 3.11, 2.47))); 
		// 14  Present Box Procedural Textured Red
		materials.push_back(new PresentBox(vec3(0.5, 0, 0), vec3(0.5, 0, 0)));

		CreateChristmasTree();
		createSnowman(vec3(1,-0.6,-0.5));
		createIcicles();
		createOranges();
		decorateTree();
		createPresent(vec3(-0.6, -0.8, -0.2), vec3(0.2, 0.2, 0.2));

		objects.push_back(new Plane(vec3(0, -1, 0), vec3(0, 1, 0), materials[11]));
	}

	~Scene()
	{
		for (std::vector<Material*>::iterator iMaterial = materials.begin(); iMaterial != materials.end(); ++iMaterial)
			delete *iMaterial;
		for (std::vector<Intersectable*>::iterator iObject = objects.begin(); iObject != objects.end(); ++iObject)
			delete *iObject;		
		for (std::vector<LightSource*>::iterator iLightSource = lightSources.begin(); 
			iLightSource != lightSources.end(); ++iLightSource)
			delete *iLightSource;

	}

	bool isShadow(LightSource* light, std::vector<Intersectable*> objects, Hit h) {
		vec3 origin = h.position + h.normal * 0.1;
		Ray ray(origin, -light->getLightDirAt(origin));

		float tMin = light->getDistanceFrom(origin);
		for (int i = 0; i < objects.size(); i++) {
			Hit hit = objects[i]->intersect(ray);

			if (hit.t > 0 && hit.t < tMin)
				tMin = hit.t;
		}

		return tMin != light->getDistanceFrom(origin);
	}

	vec3 getSkyColor(vec3 viewDir) {
		int scale = 32;
		int turbulence = 50;
		int period = 25;
		int sharpness = 1;

		float w = viewDir.x * period + pow(snoise(viewDir * scale), sharpness)*turbulence;
		w = sin(w) * 0.5 + 0.5;
		return (((vec3(0,0.667,0.333)) * w + (vec3(0.1667, 0.4166, 0.4166)) * (1 - w))) * 1.8;

	}

public:
	Camera& getCamera()
	{
		return camera;
	}

	int sign(vec3 normal, vec3 inDir) {
		float cosVal = normal.dot(inDir);
		if (cosVal < 0)
			return -1;
		else return 1;
	}

	vec3 trace(const Ray& ray, int depth)
	{
		float tMin = FLT_MAX;
		Hit bestHit;
		Hit hit;
		for (int i = 0; i < objects.size(); i++) {
			hit = objects[i]->intersect(ray);

			if (hit.t > 0 && hit.t < tMin) {
				EvergreenTree* egMat = dynamic_cast<EvergreenTree*>(hit.material);
				if (egMat) {
					if (!keepHit())
						continue;	
				}
				tMin = hit.t;
				bestHit = hit;
			}
		}

		if (tMin == FLT_MAX) {
			return getSkyColor(ray.dir);
		}

		vec3 shade(0, 0, 0);
		vec3 contribution(0, 0, 0);

		WoodenFloorGlossy* woodenFloorMat = dynamic_cast<WoodenFloorGlossy*>(bestHit.material);
		if (woodenFloorMat) {
			WoodenFloorGlossy::Event wfe = woodenFloorMat->evaluateEvent(
				ray.dir,
				bestHit.normal);
			contribution += trace(Ray(bestHit.position + bestHit.normal * 0.1, wfe.reflectionDir), depth + 1);
			shade = (contribution * wfe.reflectance) * 0.1;
		}

		Metal* metalMat = dynamic_cast<Metal*>(bestHit.material);
		if (metalMat != NULL && depth < 5) {
			Metal::Event metalEvent = metalMat->evaluateEvent(
				ray.dir,
				bestHit.normal);
			contribution += trace(Ray(bestHit.position + bestHit.normal * 0.1, metalEvent.reflectionDir), depth + 1);
			return contribution * metalEvent.reflectance;
			printf("FOUND A METAL");
		}

		Dielectric* dielectricMat = dynamic_cast<Dielectric*>(bestHit.material);
		if (dielectricMat != NULL && depth < 5) {
			Dielectric::Event de = dielectricMat->evaluateEvent(
				ray.dir,
				bestHit.normal);
			return trace(Ray(bestHit.position + bestHit.normal * 0.01 * sign(bestHit.normal, -ray.dir),
				de.reflectionDir), depth + 1) * de.reflectance +
				trace(Ray(bestHit.position - bestHit.normal * 0.001 * sign(bestHit.normal, -ray.dir),
					de.refractionDir), depth + 1) * de.transmittance;
		}

		for (int i = 0; i < lightSources.size(); i++) {
			if (!isShadow(lightSources[i], objects, bestHit))
				shade += bestHit.material->shade(
					bestHit.position, 
					bestHit.normal, 
					-ray.dir, 
					-lightSources[i]->getLightDirAt(bestHit.position), 
					lightSources[i]->getPowerDensityAt(bestHit.position));
			}
		return shade;
	}
};


Scene scene;

bool computeImage()
{
	static unsigned int iPart = 0;

	if (iPart >= 64)
		return false;
	for (int j = iPart; j < windowHeight; j += 64)
	{
		for (int i = 0; i < windowWidth; i++)
		{
			float ndcX = (2.0 * i - windowWidth) / windowWidth;
			float ndcY = (2.0 * j - windowHeight) / windowHeight;
			Camera& camera = scene.getCamera();
			Ray ray = Ray(camera.getEye(), camera.rayDirFromNdc(ndcX, ndcY));

			image[j*windowWidth + i] = scene.trace(ray, 0);
		}
	}
	iPart++;
	return true;
}

void onDisplay() {
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (computeImage())
		glutPostRedisplay();
	glDrawPixels(windowWidth, windowHeight, GL_RGB, GL_FLOAT, image);

	glutSwapBuffers();
}



int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);
	glutInitWindowPosition(100, 100);
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow("Ray Casting");

#if !defined(__APPLE__)
	glewExperimental = true;
	glewInit();
#endif

	srand(time(NULL));

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	glViewport(0, 0, windowWidth, windowHeight);

	glutDisplayFunc(onDisplay);

	glutMainLoop();

	return 1;
}
