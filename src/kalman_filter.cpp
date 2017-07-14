#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    /* predict the state */
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    /* update the state by using Kalman Filter equations */
	VectorXd z_pred = H_ * x_;
	VectorXd y      = z - z_pred;
	MatrixXd Ht     = H_.transpose();
	MatrixXd S      = H_ * P_ * Ht + R_;
	MatrixXd Si     = S.inverse();
	MatrixXd PHt    = P_ * Ht;
	MatrixXd K      = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    /* update the state by using Extended Kalman Filter equations */
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    VectorXd h(3);

    float c1 = sqrt(pow(px, 2) + pow(py, 2));
    float c2 = (px*vx)+(py*vy);
    if(fabs(c1) < 0.0001 || isinf(c1)) {
        if(c1 > 0) {
            c1 =  0.0001;
        } else {
            c1 = -0.0001;
        }
    }

    h << c1, atan2(py, px), c2/c1;

	VectorXd y = z - h;
    while(y(1) < -M_PI || M_PI < y(1)) {
        if(y(1) < -M_PI) {
            y(1) += M_PI * 2;
        } else {
            y(1) -= M_PI * 2;
        }
    }
	MatrixXd Ht  = H_.transpose();
	MatrixXd S   = H_ * P_ * Ht + R_;
	MatrixXd Si  = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K   = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
