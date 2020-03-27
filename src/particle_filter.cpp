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
     * Initializes the particle vector with num_particles .
     * It uses Gaussian Distribution with given standard deviation to add random noise.
     *
     * Initializes the weight vector of size num_particles to 1.0. This is further used in determining probability.
     */

    std::default_random_engine gen;

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    num_particles = 100;  // TODO: Set the number of particles

    particles = vector<Particle>(num_particles);
    weights = vector<double >(num_particles, 1.0);

    for( int i = 0; i < num_particles; ++i){

        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles[i] = p;
    }

    /**
     * Uncomment below lines to print all values of particles while debugging
     * */


    //for (int i =0; i < num_particles; ++i){
    //   std::cout<<" ID :"<<particles[i].id<<" X :"<<particles[i].x<<" Y :"<<particles[i].y<<" Theta :"<<particles[i].theta<<" Weight :"<<particles[i].weight<<" Weight :"<<weights[i]<<std::endl;
    //}

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * Predicts the vehicle's next position , assuming bicycle motion model.
     * Adds uncertainity using gaussian distribution to predicted position
     *
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */

    double x, y , theta;

    for(int i = 0; i < num_particles; ++i){

        //Particle p;

        if(fabs(yaw_rate) > 0.0001 ){
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
     * Used Nearest neighbour to associate Map Landmarks to Predicted Landmarks.
     */

    for(LandmarkObs observation: observations) {
        double minDistance = 9999;
        int k = -1;
        for (LandmarkObs pred: predicted) {
            double min_2;
            min_2 = dist(pred.x, pred.y, observation.x, observation.y);

            if (minDistance  > min_2) {
                minDistance = min_2;
                k = pred.id;
            }
        }
        if(k != -1){
            observation.id = k;
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * 1. Predicts landmarks within sensor range
     * 2. Transforms the position of observed landmarks which is calculated through car's perspective and
     * transforms to map coordinates with help of particle's position. This is done for each particle.
     * 3. Update weights of the particle i.e. probability using Multivariate Gaussian Distribution.
     */


    for ( int p =0; p < num_particles; ++p) {

        double p_x = particles[p].x;
        double p_y = particles[p].y;
        double p_theta = particles[p].theta;

        vector<LandmarkObs> predicted;

        for (auto land_mark : map_landmarks.landmark_list) {

            LandmarkObs predict_temp;
            //double distance = dist( particles[p].x, particles[p].y, land_mark.x_f, land_mark.y_f);

            if (dist(p_x, p_y, land_mark.x_f, land_mark.y_f) <= sensor_range ){
                predict_temp.x = land_mark.x_f;
                predict_temp.y = land_mark.y_f;
                predict_temp.id = land_mark.id_i;

                predicted.push_back(predict_temp);
            }
        }


        vector<LandmarkObs> transformed_observations;

        for (const auto & observation : observations){

            LandmarkObs t_obs;
            t_obs.x = observation.x * cos(p_theta) -  observation.y * sin(p_theta) + p_x;
            t_obs.y = observation.x * sin(p_theta) + observation.y * cos(p_theta) + p_y;
            t_obs.id = observation.id;

            transformed_observations.push_back(t_obs);
        }

        dataAssociation(predicted, transformed_observations);



        double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

        double prob = 1.0;
        for(unsigned int k = 0; k < transformed_observations.size(); ++k){

            double t_obs_x = transformed_observations[k].x;
            double t_obs_y = transformed_observations[k].y;
            int t_obs_id = transformed_observations[k].id;

            for (auto & pred : predicted ) {

                double pred_x = pred.x;
                double pred_y = pred.y;
                int pred_id = pred.id;

                if (t_obs_id == pred_id){
                    double exponent;
                    exponent = pow(pred_x - t_obs_x, 2) / (2 * pow(std_landmark[0], 2)) +
                               pow(pred_y - t_obs_y, 2) / (2 * pow(std_landmark[1], 2));
                    prob = prob * gauss_norm * exp(-exponent);
                    break;
                }
            }
        }

        particles[p].weight *=  prob;

    }

    /**
     * Calculate Normalizer  and Normalize the probability*/

    double weight_normalizer = std::accumulate(weights.begin(), weights.end(),0.0f);

    for (int p = 0; p < num_particles; ++p){
        particles[p].weight = particles[p].weight/ weight_normalizer;
        weights[p] = particles[p].weight;
    }

}

void ParticleFilter::resample() {
    /**
     * Resampling the Particles based on their Probability . Technically drawing the particles based on their probability with replacement.
     *
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */

    vector<Particle> resampled_Set;

    std::default_random_engine gen;
    std::discrete_distribution <int> sample(weights.begin(), weights.end());

    for ( int i = 0; i < num_particles; ++i){
        int k = sample(gen);
        resampled_Set.push_back(particles[k]);
    }

    particles = resampled_Set;




    /////////////////////////////////// Random Sampling Implementation///////////////////////

    /**
     * Uncomment below lines and comment above implementation to check Random sampling implementation*/

    //vector<Particle> resampled_particles;
    //std::default_random_engine gen;

    //std::uniform_int_distribution<int> particle_index(0, num_particles - 1);

    //int current_index = particle_index(gen);

    //double beta = 0.0;

    //double max_weight = *max_element(std::begin(weights), std::end(weights));

    //for (int i = 0; i < particles.size(); i++) {
     //   std::uniform_real_distribution<double> random_weight(0.0, max_weight * 2);
      //  beta += random_weight(gen);

      //  while (beta > weights[current_index]) {
      //      beta -= weights[current_index];
      //      current_index = (current_index + 1) % num_particles;
      //  }
      //  resampled_particles.push_back(particles[current_index]);
    //}

    //particles = resampled_particles;

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