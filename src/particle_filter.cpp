/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "map.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) > 0.001) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta = particles[i].theta + yaw_rate * delta_t;
		} else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
		// with noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	// observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	// implement this method and use it as a helper during the updateWeights phase.

	if (predicted.size() == 0 || observations.size() == 0) {
		return;
	}

	for (int i = 0; i < observations.size(); i++) {

		double min_dist = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
		LandmarkObs min_pred = predicted[0];

		for (int j = 1; j < predicted.size(); j++) {
			LandmarkObs pred = predicted[j];
			double curr_dist = dist(observations[i].x, observations[i].y, pred.x, pred.y);
			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				min_pred = pred;
			}
		}
		observations[i].id = min_pred.id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		vector<LandmarkObs> predicted;
		vector<LandmarkObs> transformed_observations;
		Particle p = particles[i];

		// get landmarks within sensor range
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			Map::single_landmark_s l = map_landmarks.landmark_list[j];
			if (dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range) {
				predicted.push_back(LandmarkObs({l.id_i, l.x_f, l.y_f}));
			}
		}

		// transform observations into map coordinates
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs obs = observations[j];
			double trans_x = p.x + cos(p.theta) * obs.x - sin(p.theta) * obs.y;
			double trans_y = p.y + sin(p.theta) * obs.x + cos(p.theta) * obs.y;
			transformed_observations.push_back(LandmarkObs({obs.id, trans_x, trans_y}));
		}

		dataAssociation(predicted, transformed_observations);

		// compute weights
		particles[i].weight = 1.0;
		for (int j = 0; j < transformed_observations.size(); j++) {
			LandmarkObs obs = transformed_observations[j];
			LandmarkObs pred = *find_if(predicted.begin(), predicted.end(), [obs](LandmarkObs o) { return o.id == obs.id; });

			double std_x = std_landmark[0];
			double std_y = std_landmark[1];

			double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);
			double guass_exp = pow(obs.x - pred.x, 2) / 2.0 / pow(std_x, 2) + pow(obs.y - pred.y, 2) / 2.0 / pow(std_y, 2);
			double w = gauss_norm * exp(-guass_exp);
			particles[i].weight *= w;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;
	weights.clear();

	for (int i=0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	default_random_engine gen;
	discrete_distribution<int> d(weights.begin(), weights.end());

	for (int i=0; i < num_particles; i++) {
		Particle p = particles[d(gen)];
		resampled_particles.push_back(p);
	}
	particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
