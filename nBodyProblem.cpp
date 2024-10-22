#include <numeric>
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <omp.h>

struct Particle
{
	float x, y;

	Particle operator+ (const Particle& other) {
		return { x + other.x,y + other.y };
	}

	Particle& operator+=(const Particle& other) {
		x += other.x;
		y += other.y;
		return *this;
	}
};

using Particles = std::vector<Particle>;

const float G = 6.67e-11;

float float_rand(float a, float b) {
	return a + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (b - a)));
}

Particles calc_forces_non_parallel(Particles& coords, std::vector<float> masses) {
	Particles forces(coords.size(), { 0,0 });

	size_t n = coords.size();
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			float dx = coords[i].x - coords[j].x;
			float dy = coords[i].y - coords[j].y;

			float dist = std::sqrt(dx * dx + dy * dy);
			float magnitude = (G * masses[i] * masses[j]) / (dist * dist);

			float dirX = coords[j].x - coords[i].x;
			float dirY = coords[j].y - coords[i].y;

			forces[i].x += magnitude * dirX / dist;
			forces[i].y += magnitude * dirY / dist;

			forces[j].x -= magnitude * dirX / dist;
			forces[j].y -= magnitude * dirY / dist;
		}
	}

	return forces;
}

Particles calc_forces_parallel_1(Particles& coords, std::vector<float> masses) {
	Particles forces(coords.size(), { 0,0 });

	size_t n = coords.size();
	#pragma omp parallel for
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			float dx = coords[i].x - coords[j].x;
			float dy = coords[i].y - coords[j].y;

			float dist = std::sqrt(dx * dx + dy * dy);
			float magnitude = (G * masses[i] * masses[j]) / (dist * dist);

			float dirX = coords[j].x - coords[i].x;
			float dirY = coords[j].y - coords[i].y;
			#pragma omp critical
			{
				forces[i].x += magnitude * dirX / dist;
				forces[i].y += magnitude * dirY / dist;

				forces[j].x -= magnitude * dirX / dist;
				forces[j].y -= magnitude * dirY / dist;
			}
		}
	}

	return forces;
}

void calc_forces_parallel_2(Particles& coords, std::vector<float> masses, Particles& forces) {

	size_t n = coords.size();
	#pragma omp for schedule(dynamic, 6) nowait
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			float dx = coords[i].x - coords[j].x;
			float dy = coords[i].y - coords[j].y;

			float dist = std::sqrt(dx * dx + dy * dy);
			float magnitude = (G * masses[i] * masses[j]) / (dist * dist);

			float dirX = coords[j].x - coords[i].x;
			float dirY = coords[j].y - coords[i].y;

			#pragma omp atomic
			forces[i].x += magnitude * dirX / dist;
			#pragma omp atomic
			forces[i].y += magnitude * dirY / dist;
			#pragma omp atomic
			forces[j].x -= magnitude * dirX / dist;
			#pragma omp atomic
			forces[j].y -= magnitude * dirY / dist;
		}
	}
}

void calc_forces_parallel_3(Particles& coords, std::vector<float> masses, Particles& forces) {

	size_t n = coords.size();
	#pragma omp for schedule(dynamic, 6) nowait
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) continue;

			float dx = coords[i].x - coords[j].x;
			float dy = coords[i].y - coords[j].y;

			float dist = std::sqrt(dx * dx + dy * dy);
			float magnitude = (G * masses[i] * masses[j]) / (dist * dist);

			float dirX = coords[j].x - coords[i].x;
			float dirY = coords[j].y - coords[i].y;

			forces[i].x += magnitude * dirX / dist;
			forces[i].y += magnitude * dirY / dist;
		}
	}
}

void move_particles_non_parallel (Particles& coords, Particles& forces, Particles& velocities, std::vector<float>& masses, float dt) {
	int n = coords.size();
	for (int i = 0; i < n; i++) {
		Particle dv{ forces[i].x / masses[i] * dt, forces[i].y / masses[i] * dt };
		Particle dc{ (velocities[i].x + dv.x / 2) * dt, (velocities[i].y + dv.y / 2) * dt };

		velocities[i] += dv;
		coords[i] += dc;
	}
}

void move_particles_parallel_2(Particles& coords, Particles& forces, Particles& velocities, std::vector<float>& masses, float dt) {
	int n = coords.size();
	#pragma omp for nowait
	for (int i = 0; i < n; i++) {
		Particle dv{ forces[i].x / masses[i] * dt, forces[i].y / masses[i] * dt };
		Particle dc{ (velocities[i].x + dv.x / 2) * dt, (velocities[i].y + dv.y / 2) * dt };

		velocities[i] += dv;
		coords[i] += dc;
		forces[i] = { 0.0,0.0 };
	}
}

Particles calc_n_body_non_parallel(Particles coords, Particles velocities, std::vector<float> masses, float time, float dt) {
	for (double t = 0; t < time; t += dt) {
		auto forces = calc_forces_non_parallel(coords, masses);
		move_particles_non_parallel(coords, forces, velocities, masses, dt);
	}
	return coords;
}

Particles calc_n_body_parallel_1(Particles coords, Particles velocities, std::vector<float> masses, float time, float dt) {
	for (double t = 0; t < time; t += dt) {
		auto forces = calc_forces_parallel_1(coords, masses);
		move_particles_non_parallel(coords, forces, velocities, masses, dt);
	}
	return coords;
}

Particles calc_n_body_parallel_2(Particles coords, Particles velocities, Particles forces, std::vector<float> masses, float time, float dt) {
	#pragma omp parallel
	for (double t = 0; t < time; t += dt) {
		calc_forces_parallel_2(coords, masses, forces);
		#pragma omp barrier
		move_particles_parallel_2(coords, forces, velocities, masses, dt);
		#pragma omp barrier
	}
	return coords;
}

Particles calc_n_body_parallel_3(Particles coords, Particles velocities, Particles forces, std::vector<float> masses, float time, float dt) {
	#pragma omp parallel
	for (double t = 0; t < time; t += dt) {
		calc_forces_parallel_3(coords, masses, forces);
		#pragma omp barrier
		move_particles_parallel_2(coords, forces, velocities, masses, dt);
		#pragma omp barrier
	}
	return coords;
}

void result_info(Particles& coords, Particles& velocities, std::vector<float>& masses, float time, float dt) {
	//NON PARALLEL
	auto start = std::chrono::steady_clock::now();
	auto coords_non_parallel = calc_n_body_non_parallel(coords, velocities, masses, 1.0, 0.0001);
	auto end = std::chrono::steady_clock::now();
	std::cout << "non parallel sec: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << '\n';
	//

	//PARALLEL_naive
	//start = std::chrono::steady_clock::now();
	//auto coords_naive = calc_n_body_parallel_1(coords, velocities, masses, 1.0, 0.0001);
	//end = std::chrono::steady_clock::now();
	//std::cout << "naive parallel (omp critical) sec: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << '\n';
	

	//PARALLEL_2
	start = std::chrono::steady_clock::now();
	Particles forces(coords.size(), {0.0,0.0});
	auto coords2 = calc_n_body_parallel_2(coords, velocities, forces, masses, 1.0, 0.0001);
	end = std::chrono::steady_clock::now();
	std::cout << "parallel_2 (atomic) sec: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << '\n';
	//

	//PARALLEL_3
	start = std::chrono::steady_clock::now();
	forces = Particles(coords.size(), { 0.0,0.0 });
	auto coords3 = calc_n_body_parallel_3(coords, velocities, forces, masses, 1.0, 0.0001);
	end = std::chrono::steady_clock::now();
	std::cout << "parallel_3 (without symmetric, no mutex) sec: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << '\n';
	

	float sum_dif = 0;

	for (int i = 0; i < coords.size(); i++) {
		//sum_dif += std::sqrt(std::powf((coords_non_parallel[i].x - coords_naive[i].x), 2.0) + std::powf((coords_non_parallel[i].y - coords_naive[i].y), 2.0));
		sum_dif += std::sqrt(std::powf((coords_non_parallel[i].x - coords2[i].x), 2.0) + std::powf((coords_non_parallel[i].y - coords2[i].y), 2.0));
		sum_dif += std::sqrt(std::powf((coords_non_parallel[i].x - coords3[i].x), 2.0) + std::powf((coords_non_parallel[i].y - coords3[i].y), 2.0));
	}

	std::cout <<"Sum coordinate distance difference of algirithms: " << sum_dif << '\n';
}

int main(int argc, char* argv[]) {
	srand(static_cast <unsigned> (time(0)));

	int N = 1000;//Num particles

	std::vector<Particle> coords;
	std::vector<Particle> velocities;
	std::vector<float> masses;

	for (int i = 0; i < N; i++) {
		coords.push_back({ float_rand(25,75),float_rand(25,75) });
		velocities.push_back({ float_rand(-1,1),float_rand(-1,1) });
		masses.push_back(float_rand(10000, 10000000));
	}

	result_info(coords, velocities, masses, 1.0, 0.00001);
}