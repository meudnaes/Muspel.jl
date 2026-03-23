using Muspel
using Base.Threads
using AtomicData
using BifrostTools
using ProgressMeter

"""
function calc_fe(
    xp::BifrostExperiment,
    snap::Integer,
    fe::AtomicModel;
    slicex::AbstractVector{<:Integer}=Int[], 
    slicey::AbstractVector{<:Integer}=Int[],
    slicez::AbstractVector{<:Integer}=Int[],
    verbose::Bool=false
)

Function to calculate Fe I 617 nm disk-centre intensities from Bifrost snapshot.
"""
function calc_fe(
    xp::BifrostExperiment,
    snap::Integer,
    fe::AtomicModel;
    slicex::AbstractVector{<:Integer}=Int[], 
    slicey::AbstractVector{<:Integer}=Int[],
    slicez::AbstractVector{<:Integer}=Int[],
    verbose::Bool=false
)
    my_line = fe.lines[3]  # 617.3 nm line, nλ = 74
    fe_abund = get_solar_abundances()[:Fe]

    # Take a slice and convert to meter
    z = xp.mesh.z[slicez] .* 1f6
    
    if verbose
        println("--- Loading snapshot variables ---")
    end
    # <read mesh, T, vz, n_e, rho>
    rho = get_var(xp,snap,"r",units="si",slicex=slicex,slicey=slicey,slicez=slicez)
    electron_density = get_electron_density(xp,snap,units="si",slicex=slicex,slicey=slicey,slicez=slicez,verbose=verbose)
    temperature = get_var(xp,snap,"tg",units="si",slicex=slicex,slicey=slicey,slicez=slicez)
    # Load momentum and divide by rho to get velocity
    v_z = get_var(xp,snap,"pz",units="si",slicex=slicex,slicey=slicey,slicez=slicez,destagger=true)
    v_z ./= rho

    # Permute dims so that (x, y, z) -> (z, y, x)
    new_dims = (3,2,1)
    rho = permutedims(rho, new_dims)
    temperature = permutedims(temperature, new_dims)
    v_z = permutedims(v_z, new_dims)
    electron_density = permutedims(electron_density, new_dims)

    if verbose
        println("--- Computing populations ---")
    end

    grph = 2.380491f-27
    n_H = rho ./ grph
    
    hydrogen1_density = similar(n_H)
    proton_density = similar(n_H)
    n_u = similar(n_H)
    n_l = similar(n_H)

    # compute H I and H II densities, and level populations
    @threads for i in eachindex(temperature)
        ionfrac = Muspel.h_ionfrac_saha(temperature[i], electron_density[i])
        proton_density[i] = n_H[i] * ionfrac
        hydrogen1_density[i] = n_H[i] * (1 - ionfrac)

        fe_pops = saha_boltzmann(fe, temperature[i], electron_density[i], n_H[i])
        # For this transition, upper is n=4, lower is n=1
        n_l[i] = fe_pops[1] * fe_abund
        n_u[i] = fe_pops[4] * fe_abund
    end

    atmos = Atmosphere1D(xp.mesh.mx,xp.mesh.my,length(z),Float32.(z),temperature,v_z, 
                electron_density,hydrogen1_density,proton_density)

    # Continuum opacity structures
    bckgr_atoms = [
        "Al.yaml",
        "C.yaml",
        "Ca.yaml",
        "Fe.yaml",
        "H_6.yaml",
        "He.yaml",
        "KI.yaml",
        "Mg.yaml",
        "N.yaml",
        "Na.yaml",
        "NiI.yaml",
        "O.yaml",
        "S.yaml",
        "Si.yaml",
    ]
    atom_files = [joinpath(AtomicData.get_atom_dir(), a) for a in bckgr_atoms]
    σ_itp = get_σ_itp(atmos, my_line.λ0, atom_files)

    a = LinRange(1f-4, 1f1, 20000)
    v = LinRange(0f0, 5f2, 2500)
    voigt_itp = create_voigt_itp(a, v)

    if verbose
        println("--- Calculating intensity ---")
    end

    intensity = Array{Float32, 3}(undef, my_line.nλ, atmos.ny, atmos.nx)

    p = ProgressMeter.Progress(atmos.nx)
    @threads for i in 1:atmos.nx
        buf = RTBuffer(atmos.nz, my_line.nλ, Float32)  # allocate inside for local scope
        for j in 1:atmos.ny
            calc_line_prep!(my_line, buf, atmos[:, j, i], σ_itp)
            calc_line_1D!(my_line, buf, line.λ, atmos[:, j, i], n_u[:, j, i], n_l[:, j, i], voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
        ProgressMeter.next!(p)
    end

    return intensity
end
