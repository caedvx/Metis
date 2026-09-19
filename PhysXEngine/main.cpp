#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Particle.hpp"

int main()
{
    // All quantities SI: metres, kilograms, seconds.
    const double dt = 0.001;
    const double totalTime = 3600.0;    // one full lunar orbit is 2.358e6 s
    const double printInterval = 1.0;

    const double earthMass = 5.97e24;
    const double moonMass = 7.34e22;
    const double separation = 3.844e8;

    // Circular relative speed, split so the barycentre starts at rest.
    const double vRel = std::sqrt(physics::G * (earthMass + moonMass) / separation);
    const double vEarth = vRel * moonMass / (earthMass + moonMass);
    const double vMoon = vRel * earthMass / (earthMass + moonMass);

    std::vector<Particle> particles;
    particles.reserve(10000);

    particles.push_back({ 1, {0.0, 0.0, 0.0}, {-vEarth, 0.0, 0.0}, earthMass, true});
    particles.push_back({ 2, {0.0, separation, 0.0}, {vMoon, 0.0, 0.0}, moonMass, true});

    std::vector<Vector3> forces(particles.size());

    const long printEvery = std::max(1L, std::lround(printInterval / dt));
    const double energy0 = totalEnergy(particles);

    std::cout << std::setprecision(10);
    std::cout << "orbital period = "
              << 2.0 * M_PI * std::sqrt(separation * separation * separation
                                        / (physics::G * (earthMass + moonMass)))
              << " s\n";
    std::cout << "E0 = " << energy0 << " J\n\n";

    double time = 0.0;
    long step = 0;

    while (time < totalTime)
    {
        // Pass 1: all forces from the same snapshot of positions
        for (size_t i = 0; i < particles.size(); ++i)
        {
            forces[i] = {0.0, 0.0, 0.0};
            if (!particles[i].isActive) continue;

            for (size_t j = 0; j < particles.size(); ++j)
            {
                if (i == j || !particles[j].isActive) continue;

                forces[i] += calculateGravitationalForce(particles[i], particles[j]);
            }
        }

        // Pass 2: semi-implicit Euler integration
        for (size_t i = 0; i < particles.size(); ++i)
        {
            if (!particles[i].isActive) continue;

            particles[i].velocity += applyForce(particles[i], forces[i], dt);
            particles[i].position += particles[i].velocity * dt;
        }

        ++step;
        time = step * dt;   // not time += dt: 0.001 is inexact in binary and
                            // 3.6e6 additions would accumulate visible drift

        if (step % printEvery == 0)
        {
            for (const Particle& p : particles)
            {
                std::cout << "t=" << time << " particle " << p.id
                          << " position: " << p.position
                          << " velocity: " << p.velocity << "\n";
            }

            double energy = totalEnergy(particles);
            Vector3 momentum = totalMomentum(particles);

            std::cout << "  E=" << energy
                      << " drift=" << std::abs((energy - energy0) / energy0) * 100.0 << "%"
                      << " p=" << momentum << "\n";
        }
    }

    return 0;
}
