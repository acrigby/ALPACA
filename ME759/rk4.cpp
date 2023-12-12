#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <random>
using namespace std;

void dsdt(double* deriv[], double theta1, double theta2,double dtheta1,double dtheta2,double a){

    double dt = 0.2;
    double pi = 3.14158 ; 
    double g = 9.81;

    double l1 = 1.0;  // [m]
    double l2 = 1.0;  // [m]
    double m1 = 1.0;  //: [kg] mass of link 1
    double m2 = 1.0 ; //: [kg] mass of link 2
    double lc1 = 0.5 ; //: [m] position of the center of mass of link 1
    double lc2 = 0.5 ; //: [m] position of the center of mass of link 2
    double I1 = 1.0 ; //: moments of inertia for both links
    double I2 = 1.0 ; //: moments of inertia for both links

    double d1 = (
            m1 * (lc1*lc1)
            + m2 * ((l1*l1) + (lc2*lc2) + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        );
    double d2 = m2 * ((lc2*lc2) + l1 * lc2 * cos(theta2)) + I2 ;
    double phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0);
    double phi1 = (
            -m2 * l1 * lc2 * (dtheta2*dtheta2) * sin(theta2)
            - 2 * m2 * l1 * lc2 * (dtheta2 * dtheta1) * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        );
    double ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * (dtheta1*dtheta1) * sin(theta2) - phi2
            ) / (m2 * (lc2*lc2) + I2 - (d2*d2) / d1);
    double ddtheta1 = -(d2 * ddtheta2 + phi1) / d1 ;

    (*deriv)[0] = dtheta1;
    (*deriv)[1] = dtheta2;
    (*deriv)[2] = ddtheta1;
    (*deriv)[3] = ddtheta2;
    (*deriv)[4] = 0;

}

int main(int argc, char *argv[]) {

    usleep(16000);

    double *y = new double[5];
    double *y1 = new double[5];

    y[0] = atof(argv[1]);
    y[1] = atof(argv[2]);
    y[2] = atof(argv[3]);
    y[3] = atof(argv[4]);
    y[4] = atof(argv[5]);
    
    double dt = atof(argv[6]);
    double dt2 = dt/2;

    double *k1 = new double[5];
    double *k2 = new double[5];
    double *k3 = new double[5];
    double *k4 = new double[5];

    dsdt(&k1, y[0],               y[1],              y[2],              y[3],             y[4]);
    dsdt(&k2, y[0] + dt2 * k1[0], y[1]+ dt2 * k1[1], y[2]+ dt2 * k1[2], y[3]+ dt2 * k1[3],y[4]+dt2 * k1[4]);
    dsdt(&k3, y[0] + dt2 * k2[0], y[1]+ dt2 * k2[1], y[2]+ dt2 * k2[2], y[3]+ dt2 * k2[3],y[4]+dt2 * k2[4]);
    dsdt(&k4, y[0] + dt * k3[0],  y[1]+ dt * k3[1],  y[2]+ dt * k3[2],  y[3]+ dt * k3[3], y[4]+dt * k3[4]);

    for(int i = 0; i < 5; i++){
        y1[i] = y[i] + dt/6 * (k1[i] + 2* k2[i] + 2*k3[i] + k4[i]);
    }

    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;

    cout << y1[0] <<','<< y1[1] << ','<< y1[2]<< ','<< y1[3] << ','<< y1[4]<< endl;
    cout.flush();

    delete[] y;
    delete[] y1;

    return 0;

}