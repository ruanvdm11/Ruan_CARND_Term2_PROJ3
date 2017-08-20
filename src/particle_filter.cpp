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
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//Number of particles
	num_particles = 8;

	// Distributions
	default_random_engine gen;
	normal_distribution<double> xDist(x, std[0]);
	normal_distribution<double> yDist(y, std[1]);
	normal_distribution<double> thetaDist(theta, std[2]);

	//Initialize particles
	for (int i = 0; i < num_particles; i++){
		Particle par;
		par.id = i;
		par.x = xDist(gen);
		par.y = yDist(gen);
		par.theta = thetaDist(gen);
		par.weight = 1.0;
		particles.push_back(par);
		weights.push_back(1.0);
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Prediction Distributions
	default_random_engine gen;
	normal_distribution<double> xDist(0, std_pos[0]);
	normal_distribution<double> yDist(0, std_pos[1]);
	normal_distribution<double> thetaDist(0, std_pos[2]);


	//Insure the correct motion model is used depending on yaw rate
	for (auto& par:particles){

		double yawRateDT = yaw_rate*delta_t;
		double velOverYawrate = velocity/yaw_rate;
		if(fabs(yaw_rate)>0.001){
			

			par.x += velOverYawrate*(sin(par.theta + yawRateDT) - sin(par.theta)) + xDist(gen);
			par.y += velOverYawrate*(-cos(par.theta + yawRateDT) + cos(par.theta)) + yDist(gen);
		} else{
			par.x += velocity*cos(par.theta)*delta_t + xDist(gen);
			par.y += velocity*sin(par.theta)*delta_t + yDist(gen);
		}
		par.theta += yawRateDT + thetaDist(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	Map landmarksNew;
	landmarksNew.landmark_list.clear();
	weights.clear();

	for(auto& par : particles){
		double thetaT = par.theta;

		for(int j = 0; j < map_landmarks.landmark_list.size(); j++){
			double distance = dist(par.x, par.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

			if(distance<sensor_range){
				landmarksNew.landmark_list.push_back(map_landmarks.landmark_list[j]);
			}
		}
		par.weight = 1.0;

		LandmarkObs observs;
		for(auto& observs:observations){
			//CoordinateTranslation
			double xTrans = cos(thetaT)*observs.x - sin(thetaT)*observs.y + par.x;
			double yTrans = sin(thetaT)*observs.x + cos(thetaT)*observs.y + par.y;

			//Data Association
			double minDistance = std::numeric_limits<float>::max();
			int mapId = -1;

			for(int k = 0; k < landmarksNew.landmark_list.size(); k++){
				double distCalc = dist(xTrans, yTrans, landmarksNew.landmark_list[k].x_f, landmarksNew.landmark_list[k].y_f);
				if(distCalc<minDistance){
					minDistance = distCalc;
					mapId = landmarksNew.landmark_list[k].id_i;
				}
			}
			// Weight calculations constants
			double xProb = -0.5*(xTrans-map_landmarks.landmark_list[mapId-1].x_f)*(xTrans-map_landmarks.landmark_list[mapId-1].x_f)/(std_landmark[0]*std_landmark[0]);
			double yProb = -0.5*(yTrans-map_landmarks.landmark_list[mapId-1].y_f)*(yTrans-map_landmarks.landmark_list[mapId-1].y_f)/(std_landmark[1]*std_landmark[1]);
			double probDeno = 2*M_PI*std_landmark[0]*std_landmark[1];

			par.weight *= exp((xProb + yProb)) / probDeno;
		}
		weights.push_back(par.weight);
		landmarksNew.landmark_list.clear();

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> discreteDist(weights.begin(), weights.end());
	std::map<int, int> m;

	for (int n = 0; n<num_particles; n++){
		new_particles.push_back(particles[discreteDist(gen)]);

	}
	particles = new_particles;
	weights.clear();

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
