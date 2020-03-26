/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 100;  // TODO: Set the number of particles

    for( int i = 0; i < num_particles; ++i){

        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }
    is_initialized = true;

    for (int i =0; i < num_particles; ++i){
        std::cout<<" ID :"<<particles[i].id<<" X :"<<particles[i].x<<" Y :"<<particles[i].y<<" Theta :"<<particles[i].theta<<" Weight :"<<particles[i].weight<<" Weight :"<<weights[i]<<std::endl;
    }

    std::cout<<"\n";
    std::cout<<"Initialized : "<<is_initialized<<std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

    double x, y , theta;

    for(int i = 0; i < num_particles; ++i){

        //Particle p;

        if(fabs(yaw_rate) > 0.001 ){
            x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t ) - sin(particles[i].theta)) ;
            y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t )) ;
            theta = particles[i].theta + yaw_rate * delta_t ;
        }
        else{
            x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
            y = particles[i].y + velocity * delta_t * sin(particles[i].theta) ;
            theta = particles[i].theta + yaw_rate * delta_t ;
        }


        std::default_random_engine gen;
        std::normal_distribution<double> dist_x(x, std_pos[0]);
        std::normal_distribution<double> dist_y(y, std_pos[1]);
        std::normal_distribution<double> dist_theta(theta, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);

    }


}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */

    for(LandmarkObs pred: predicted) {
        double minDistance = 9999;

        for (LandmarkObs observation: observations) {
            double min_2 = dist(pred.x, pred.y, observation.x, observation.y);

            if (minDistance  > min_2) {
                minDistance = min_2;
                observation.id = pred.id;
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a multi-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    double weight_normalizer = 0.0;

    for ( int p =0; p < num_particles; ++p) {

        vector<LandmarkObs> predicted;

        for (auto land_mark : map_landmarks.landmark_list) {

            LandmarkObs predict_temp{};
            double distance = dist( particles[p].x, particles[p].y, land_mark.x_f, land_mark.y_f);

            if ( distance < sensor_range ){
                predict_temp.x = land_mark.x_f;
                predict_temp.y = land_mark.y_f;
                predict_temp.id = land_mark.id_i;

                predicted.push_back(predict_temp);
            }
        }

        //particles[p].weight= 1.0;

        vector<LandmarkObs> transformed_observations;

        for (const auto & observation : observations){

            LandmarkObs t_obs{};
            t_obs.x = observation.x * cos(particles[p].theta) -  observation.y * sin(particles[p].theta) + particles[p].x;
            t_obs.y = observation.x * sin(particles[p].theta) + observation.y * cos(particles[p].theta) + particles[p].y;
            t_obs.id = observation.id;

            transformed_observations.push_back(t_obs);
        }

        dataAssociation(predicted, transformed_observations);



        double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

        double prob = 1.0;
        for(auto & pred : predicted){

            double pred_x = pred.x;
            double pred_y = pred.y;
            double pred_id = pred.id;

            for (unsigned int k = 0; k < transformed_observations.size(); ++k) {

                double t_obs_x = transformed_observations[k].x;
                double t_obs_y = transformed_observations[k].y;
                double t_obs_id = transformed_observations[k].id;

                if (t_obs_id == pred_id){
                    double exponent = pow(pred_x - t_obs_x, 2) / 2 * pow(std_landmark[0], 2) +
                                      pow(pred_y - t_obs_y, 2) / 2 * pow(std_landmark[1], 2);
                    prob = prob * gauss_norm * exp(-exponent);
                    break;
                }
            }
        }

        particles[p].weight =  prob;
        weights[p] = prob;
        weight_normalizer = weight_normalizer + prob;


    }

    //double weight_normalizer = std::accumulate(weights.begin(), weights.end(),0.0f);
    //double weight_normalizer = 0.0;
    //for (int i = 0; i < num_particles; ++i){
     //   weight_normalizer += weights[i];
    //}
    for (int p = 0; p < num_particles; ++p){
        particles[p].weight = particles[p].weight/ weight_normalizer;
        weights[p] = particles[p].weight;
    }

    for (int p = 0; p < num_particles; ++p){
        std::cout<<"\n";
        std::cout<<"ID :"<<p<<" Probability :"<<particles[p].weight<<" Weights :"<<weights[p];
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    //vector<float> prob ;
    vector<Particle> resampled_Set;

    std::default_random_engine gen;
    std::discrete_distribution <> sample(weights.begin(), weights.end());

    for ( int i = 0; i < num_particles; ++i){
       resampled_Set.push_back(particles[sample(gen)]);
    }

    particles = resampled_Set;


    std::cout<<"\n";
    for (int i =0; i < num_particles; ++i){
        std::cout<<" ID :"<<particles[i].id<<" X :"<<particles[i].x<<" Y :"<<particles[i].y<<" Theta :"<<particles[i].theta<<" Weight :"<<particles[i].weight<<" Weight :"<<weights[i]<<std::endl;
    }

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}