module PSO

using ..ConfigModule, ..Log

using Random

export run_sim, Best

mutable struct Best
    position::BitArray
    loss::Float64
end

mutable struct Particle
    position::BitArray # Equivalent to the genotype for a GA
    velocity::Vector{Float64} 
    loss::Float64
    pbest::Best 
end

mutable struct Swarm
    swarm_size::Int
    particles::Vector{Particle}
    gbest::Best # Will hold the index of the swarm's current best. Or is it better to just hold the actual position?
    inertia_weight::Float64 # ?
    cognitive_coeff::Float64
    social_coeff::Float64
end

# Is velocity one value? Or is it per dimension in the position?

function bit_array_to_bitstring(bitarr::BitArray)
    return replace(bitstring(bitarr), " "=>"")
end


function initialize_swarm(swarm_size::Int, position_length::Int, lookup_table::Dict{Any, Any})
    positions = [bitrand(Xoshiro(), position_length) for _ in 1:swarm_size]
    particles = []
    best = Best(positions[1], Inf)
    for position in positions
        particle_loss = lookup_table[bit_array_to_bitstring(position)]
        if particle_loss < best.loss
            best.position = position
            best.loss = particle_loss
        end
        push!(particles, Particle(position, [0.0 for _ in 1:position_length], particle_loss, Best(position, particle_loss)))
    end
    return particles, best
end

function σ(x)
    return 1/(1+ℯ^(-x))
end

function update_vel!(particle::Particle, swarm::Swarm)
    for i in 1:length(particle.velocity)
        particle.velocity[i] = σ(swarm.inertia_weight * particle.velocity[i] + swarm.cognitive_coeff * rand(0:1) * (particle.pbest.position[i] - particle.position[i]) + swarm.social_coeff * rand(0:1) * (swarm.gbest.position[i] - particle.position[i])) 
    end
end

function update_pos!(particle::Particle, swarm::Swarm)
    for i in 1:length(particle.velocity)
        if particle.velocity[i] < rand(Float64)
            particle.position[i] = 1
        else
            particle.position[i] = 0
        end
    end
end

function calc_avg_swarm_loss(swarm::Swarm)
    val = 0.0
    for particle in swarm.particles
        val += particle.loss
    end
    return val / swarm.swarm_size
end

function run_sim(config::Config, lookup_table::Dict{Any, Any})
    history = History([], [])
    particles, best = initialize_swarm(config.pop_size, config.genotype_len, lookup_table)
    swarm = Swarm(
        config.pop_size, 
        particles,
        best,
        0.9, # Usually between 0.4-0.9, so may be worth tweaking a little
        2.0,
        2.0
    )

    for _ in 1:config.num_gen # Could also add a tol threshold for changes in pbest...
        for particle in swarm.particles

            update_vel!(particle, swarm)
            update_pos!(particle, swarm)

            new_loss = lookup_table[bit_array_to_bitstring(particle.position)]
            if new_loss < particle.loss
                particle.pbest.position = particle.position
                particle.pbest.loss = new_loss
                if new_loss < swarm.gbest.loss 
                    swarm.gbest.position = particle.position
                    swarm.gbest.loss = new_loss
                end
            end
            particle.loss = new_loss
        end     
        # Record the swarm best 
        push!(history.best_global_loss, swarm.gbest.loss)
        push!(history.avg_swarm_loss, calc_avg_swarm_loss(swarm))
    end

    return swarm.gbest, history

end

end