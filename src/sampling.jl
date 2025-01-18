# helper functions for the kernel preparation
function grid_sample_validate_dimensions(input::AbstractArray{T, N}, grid, ndims) where {T,N}
    dims = ndims - 2  # Subtract channels and batch dimensions
    @assert size(grid, 1) == dims "Grid must have $dims coordinates as first dimension"
    @assert size(input, N) == size(grid, N) "Batch sizes must match"
end

function grid_sample_allocate_output(input::AbstractArray{T,N}, grid) where {T,N}
    channels = size(input, N - 1)
    batch_size = size(input, N)
    spatial_dims = ntuple(i -> size(grid, i + 1), N - 2)
    output_size = (spatial_dims..., channels, batch_size)
    similar(input, output_size)
end

"""
    grid_sample(input::AbstractArray{T, N}, grid::AbstractArray{T, N};
               interpolation_mode::String = "linear",
               padding_mode::String = "zeros",
               align_corners::Bool = false) where {T <: AbstractFloat, N}

Sample values from `input` at the coordinates specified by `grid` using interpolation.
Supports 1D (N=3), 2D (N=4), and 3D (N=5) sampling, where N includes the channel and batch dimensions.

# Arguments
- `input`: The source tensor to sample from
  - 1D: Size (W, C, N) for width, channels, batch
  - 2D: Size (W, H, C, N) for width, height, channels, batch
  - 3D: Size (W, H, D, C, N) for width, height, depth, channels, batch
- `grid`: Tensor of normalized coordinates where values will be sampled.
         Each coordinate should be in the range [-1, 1]
  - 1D: Size (1, Wout, N) for output width, batch
  - 2D: Size (2, Wout, Hout, N) for output width/height, batch
  - 3D: Size (3, Wout, Hout, Dout, N) for output width/height/depth, batch
- `interpolation_mode`: Interpolation method to use
  - "linear": Use linear interpolation (bilinear for 2D, trilinear for 3D)
  - "nearest": Use nearest neighbor interpolation
- `padding_mode`: How to handle sampling outside the input boundaries
  - "zeros": Return zero for all samples outside the input boundaries
  - "border": Use the values at the border of the input tensor
- `align_corners`: How to map the grid coordinates to input coordinates
  - `true`: The extreme values (-1, 1) are mapped to the centers of the first
           and last pixels in the input tensor
  - `false`: The extreme values (-1, 1) are mapped to the outer edges of the first
            and last pixels in the input tensor

# Example (1D)
```julia
# Create a simple 1D signal with 2 channels and batch size 1
input = reshape([1.0, 2.0, 3.0, 4.0], 4, 2, 1)  # Width=4, Channels=2, Batch=1
# Sample at points -1.0 (start) and 1.0 (end)
grid = reshape([-1.0, 1.0], 1, 2, 1)  # 1 coord, 2 output points, batch=1
result = grid_sample(input, grid, align_corners=true)
# result[:, 1, 1] ≈ [1.0, 4.0]  # First channel
# result[:, 2, 1] ≈ [2.0, 8.0]  # Second channel
```

Note: For linear interpolation, if any of the interpolation points fall outside
the input boundaries:
- In "zeros" mode: The entire interpolation result will be 0
- In "border" mode: The border values will be used for interpolation
"""
function grid_sample(input::AbstractArray{T,N},
    grid::AbstractArray{<:AbstractFloat,N};
    interpolation_mode=:linear,
    padding_mode=:zeros,
    align_corners=true) where {T<:AbstractFloat,N}

    grid_sample_validate_dimensions(input, grid, N)
    output = grid_sample_allocate_output(input, grid)

    backend = get_backend(input)

    if N == 3  # 1D
        workgroup = min(256, size(output, 1))
        ndrange = size(output, 1)
    elseif N == 4  # 2D
        workgroup = (16, 16)  # 256 threads total in 16x16 blocks
        ndrange = (size(output, 1), size(output, 2))
    elseif N == 5  # 3D
        workgroup = (8, 8, 8)  # 512 threads total in 8x8x8 blocks
        ndrange = (size(output, 1), size(output, 2), size(output, 3))
    else
        error("Unsupported dimension: $N")
    end

    kernel! = grid_sample_kernel!(backend, workgroup)
    kernel!(output, input, grid,
        Val(Symbol(interpolation_mode)),
        Val(Symbol(padding_mode)),
        align_corners,
        ndrange=ndrange)

    KernelAbstractions.synchronize(backend)
    return output
end


function ∇grid_sample(
    Δ::AbstractArray{T,N},
    input::AbstractArray{T,N},
    grid::AbstractArray{<:AbstractFloat,N};
    interpolation_mode=:linear, padding_mode=:zeros, align_corners=true) where {T,N}

    backend = get_backend(input)
    if N == 3
        workgroup = min(256, size(Δ, 1))
        ndrange = size(Δ, 1)
    elseif N == 4
        workgroup = (16, 16)  # 256 threads total in 16x16 blocks
        ndrange = (size(Δ, 1), size(Δ, 2))
    elseif N == 5
        workgroup = (8, 8, 8)  # 512 threads total in 8x8x8 blocks
        ndrange = (size(Δ, 1), size(Δ, 2), size(Δ, 3))
    else
        error("Unsupported dimension: $N")
    end

    dx = zero(input)
    dgrid = similar(grid)

    kernel! = ∇grid_sample_kernel!(backend, workgroup)
    kernel!(dx, dgrid, Δ, input, grid,
        Val(Symbol(interpolation_mode)),
        Val(Symbol(padding_mode)),
        align_corners,
        ndrange=ndrange)

    KernelAbstractions.synchronize(backend)

    return dx, dgrid
end

function rrule(::typeof(grid_sample), x, grid; interpolation_mode, padding_mode, align_corners)
    y = grid_sample(x, grid; interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
    function grid_sample_pullback(Δ)
        ∇x, ∇grid = ∇grid_sample(unthunk(Δ), x, grid; padding_mode=padding_mode)
        NoTangent(), ∇x, ∇grid
    end
    return y, grid_sample_pullback
end


# kernel helper functions
@inline in_bounds(w, W) = 1 ≤ w ≤ W
@inline in_bounds(w, h, W, H) = 1 ≤ w ≤ W && 1 ≤ h ≤ H
@inline in_bounds(w, h, d, W, H, D) = 1 ≤ w ≤ W && 1 ≤ h ≤ H && 1 ≤ d ≤ D

# might in future depend on align_corners for other padding modes
@inline function clip_coordinate(coord::T, size, padding, align_corners) where {T}
    if padding == :border
        return clamp(coord, T(1), T(size))
    else
        return coord
    end
end

@inline function ∇clip_coordinate(coord::T, size, padding, align_corners) where {T}
    if padding == :border
        if coord <= one(T)
            return T(1), T(0)
        elseif coord >= size
            return T(size), T(0)
        else
            return coord, T(1)
        end
    else
        return coord, T(1)
    end
end

@inline function unnormalize(coord::T, size, align_corners) where {T}
    if align_corners
        # unnormalize coord from [-1, 1] to [1, size] for Julia's 1-based indexing
        return T(0.5) * (coord + T(1)) * (size - T(1)) + T(1)
    else
        # unnormalize coord from [-1, 1] to [0.5, size + 0.5] for Julia's 1-based indexing
        return T(0.5) * (coord * size + size + one(T))
    end
end

@inline function ∇unnormalize(coord::T, size, align_corners) where {T}
    # constant propagation should elide the double if condition
    return unnormalize(coord, size, align_corners), align_corners ? T(0.5) * (size - 1) : T(0.5) * size
end

@inline function compute_source_index(coord, size, padding::Symbol, align_corners)
    coord = unnormalize(coord, size, align_corners)
    clipped = clip_coordinate(coord, size, padding, align_corners)
    return clipped
end

@inline function ∇compute_source_index(coord, size, padding::Symbol, align_corners)
    source_coordinate, grad_in = ∇unnormalize(coord, size, align_corners)
    source_coordinate, grad_clip = ∇clip_coordinate(source_coordinate, size, padding, align_corners)
    return source_coordinate, grad_in * grad_clip
end


###################
# Forward kernels #
###################


@kernel function grid_sample_kernel!(output::AbstractArray{T,3},
    @Const(input::AbstractArray{T,3}),
    @Const(grid::AbstractArray{<:AbstractFloat,3}),
    ::Val{interpolation},
    ::Val{padding},
    align_corners::Bool
) where {T,interpolation,padding}

    @uniform in_width, channels, batch = size(input)
    w = @index(Global, Linear)

    @inbounds for n in 1:batch
        x = grid[1, w, n]
        ix = compute_source_index(x, in_width, padding, align_corners)

        if interpolation == :linear
            ix_w = unsafe_trunc(Int, floor(ix))
            ix_e = ix_w + 1
            w_weight = ix_e - ix  # west weight
            e_weight = ix - ix_w  # east weight

            for c in 1:channels
                val = zero(T)
                if in_bounds(ix_w, in_width)
                    val += input[ix_w, c, n] * w_weight
                end
                if in_bounds(ix_e, in_width)
                    val += input[ix_e, c, n] * e_weight
                end

                output[w, c, n] = val
            end
        elseif interpolation == :nearest
            ix_nearest = unsafe_trunc(Int, round(ix))
            for c in 1:channels
                output[w, c, n] = in_bounds(ix_nearest, in_width) ? input[ix_nearest, c, n] : zero(T)
            end
        end
    end
end

@kernel function grid_sample_kernel!(output::AbstractArray{T,4},
    @Const(input::AbstractArray{T,4}),
    @Const(grid::AbstractArray{<:AbstractFloat,4}),
    ::Val{interpolation},
    ::Val{padding},
    align_corners::Bool
) where {T,interpolation,padding}

    @uniform iH, iW, channels, batch = size(input)
    w, h = @index(Global, NTuple)

    @inbounds for n in 1:batch
        x = grid[1, w, h, n]
        y = grid[2, w, h, n]
        ix = compute_source_index(x, iW, padding, align_corners)
        iy = compute_source_index(y, iH, padding, align_corners)

        if interpolation == :linear
            ix_nw, iy_nw = unsafe_trunc(Int, floor(ix)), unsafe_trunc(Int, floor(iy))
            ix_ne, iy_ne = ix_nw + 1, iy_nw
            ix_sw, iy_sw = ix_nw, iy_nw + 1
            ix_se, iy_se = ix_ne, iy_sw

            nw = (ix_se - ix) * (iy_se - iy)
            ne = (ix - ix_sw) * (iy_sw - iy)
            sw = (ix_ne - ix) * (iy - iy_ne)
            se = (ix - ix_nw) * (iy - iy_nw)

            for c in 1:channels
                val = zero(T)
                if in_bounds(iy_nw, ix_nw, iH, iW)
                    val += input[ix_nw, iy_nw, c, n] * nw
                end
                if in_bounds(iy_ne, ix_ne, iH, iW)
                    val += input[ix_ne, iy_ne, c, n] * ne
                end
                if in_bounds(iy_sw, ix_sw, iH, iW)
                    val += input[ix_sw, iy_sw, c, n] * sw
                end
                if in_bounds(iy_se, ix_se, iH, iW)
                    val += input[ix_se, iy_se, c, n] * se
                end
                output[w, h, c, n] = val
            end
        elseif interpolation == :nearest
            ix_nearest = round(Int, ix)
            iy_nearest = round(Int, iy)
            for c in 1:channels
                output[w, h, c, n] = in_bounds(iy_nearest, ix_nearest, iH, iW) ? input[ix_nearest, iy_nearest, c, n] : zero(T)
            end
        end
    end
end

@kernel function grid_sample_kernel!(output::AbstractArray{T,5},
    @Const(input::AbstractArray{T,5}),
    @Const(grid::AbstractArray{<:AbstractFloat,5}),
    ::Val{interpolation},
    ::Val{padding},
    align_corners::Bool
) where {T,interpolation,padding}

    @uniform in_width, in_height, in_depth, channels, batch = size(input)  # Changed order to WHDCN
    w, h, d = @index(Global, NTuple{3,Int})

    @inbounds for n in 1:batch
        x = grid[1, w, h, d, n]  # width coordinate
        y = grid[2, w, h, d, n]  # height coordinate
        z = grid[3, w, h, d, n]  # depth coordinate

        ix = compute_source_index(x, in_width, padding, align_corners)
        iy = compute_source_index(y, in_height, padding, align_corners)
        iz = compute_source_index(z, in_depth, padding, align_corners)

        if interpolation == :linear
            # Get the top-north-west corner
            ix_tnw = unsafe_trunc(Int, floor(ix))
            iy_tnw = unsafe_trunc(Int, floor(iy))
            iz_tnw = unsafe_trunc(Int, floor(iz))

            # Define all corners based on tnw
            # top-north-east
            ix_tne = ix_tnw + 1
            iy_tne = iy_tnw
            iz_tne = iz_tnw

            # top-south-west
            ix_tsw = ix_tnw
            iy_tsw = iy_tnw + 1
            iz_tsw = iz_tnw

            # top-south-east
            ix_tse = ix_tnw + 1
            iy_tse = iy_tnw + 1
            iz_tse = iz_tnw

            # bottom-north-west
            ix_bnw = ix_tnw
            iy_bnw = iy_tnw
            iz_bnw = iz_tnw + 1

            # bottom-north-east
            ix_bne = ix_tnw + 1
            iy_bne = iy_tnw
            iz_bne = iz_tnw + 1

            # bottom-south-west
            ix_bsw = ix_tnw
            iy_bsw = iy_tnw + 1
            iz_bsw = iz_tnw + 1

            # bottom-south-east
            ix_bse = ix_tnw + 1
            iy_bse = iy_tnw + 1
            iz_bse = iz_tnw + 1

            # Calculate interpolation weights
            tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
            tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
            tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
            tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
            bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
            bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
            bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
            bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

            for c in 1:channels
                val = zero(T)

                # Apply weights for all 8 corners
                if in_bounds(ix_tnw, iy_tnw, iz_tnw, in_width, in_height, in_depth)
                    val += input[ix_tnw, iy_tnw, iz_tnw, c, n] * tnw
                end
                if in_bounds(ix_tne, iy_tne, iz_tne, in_width, in_height, in_depth)
                    val += input[ix_tne, iy_tne, iz_tne, c, n] * tne
                end
                if in_bounds(ix_tsw, iy_tsw, iz_tsw, in_width, in_height, in_depth)
                    val += input[ix_tsw, iy_tsw, iz_tsw, c, n] * tsw
                end
                if in_bounds(ix_tse, iy_tse, iz_tse, in_width, in_height, in_depth)
                    val += input[ix_tse, iy_tse, iz_tse, c, n] * tse
                end
                if in_bounds(ix_bnw, iy_bnw, iz_bnw, in_width, in_height, in_depth)
                    val += input[ix_bnw, iy_bnw, iz_bnw, c, n] * bnw
                end
                if in_bounds(ix_bne, iy_bne, iz_bne, in_width, in_height, in_depth)
                    val += input[ix_bne, iy_bne, iz_bne, c, n] * bne
                end
                if in_bounds(ix_bsw, iy_bsw, iz_bsw, in_width, in_height, in_depth)
                    val += input[ix_bsw, iy_bsw, iz_bsw, c, n] * bsw
                end
                if in_bounds(ix_bse, iy_bse, iz_bse, in_width, in_height, in_depth)
                    val += input[ix_bse, iy_bse, iz_bse, c, n] * bse
                end

                output[w, h, d, c, n] = val
            end
        elseif interpolation == :nearest
            ix_nearest = unsafe_trunc(Int, round(ix))
            iy_nearest = unsafe_trunc(Int, round(iy))
            iz_nearest = unsafe_trunc(Int, round(iz))

            for c in 1:channels
                if in_bounds(ix_nearest, iy_nearest, iz_nearest, in_width, in_height, in_depth)
                    output[w, h, d, c, n] = input[ix_nearest, iy_nearest, iz_nearest, c, n]
                else
                    output[w, h, d, c, n] = zero(T)
                end
            end
        end
    end
end


####################
# Backward kernels #
####################


@kernel function ∇grid_sample_kernel!(
    dx::AbstractArray{T,3},
    dgrid::AbstractArray{T,3},
    @Const(Δ::AbstractArray{T,3}),
    @Const(input::AbstractArray{T,3}),
    @Const(grid::AbstractArray{<:AbstractFloat,3}),
    ::Val{interpolation},
    ::Val{padding},
    align_corners::Bool
) where {T,interpolation,padding}

    @uniform iW, channels, batch = size(input)
    w = @index(Global)

    @inbounds for n in 1:batch
        x = grid[1, w, n]
        ix, gix_mult = ∇compute_source_index(x, iW, padding, align_corners)

        if interpolation == :linear
            ix_w = unsafe_trunc(Int, floor(ix))
            ix_e = ix_w + 1
            w_weight = ix_e - ix
            e_weight = ix - ix_w

            gix = 0.0f0
            for c in 1:channels
                g_out = Δ[w, c, n]

                if in_bounds(ix_w, iW)
                    Atomix.@atomic dx[ix_w, c, n] += g_out * w_weight
                    # dx[ix_w, c, n] += g_out * w_weight
                    w_val = input[ix_w, c, n]
                    gix -= w_val * g_out
                end
                if in_bounds(ix_e, iW)
                    Atomix.@atomic dx[ix_e, c, n] += g_out * e_weight
                    # dx[ix_e, c, n] += g_out * e_weight
                    e_val = input[ix_e, c, n]
                    gix += e_val * g_out
                end
            end

            dgrid[1, w, n] = gix_mult * gix

        elseif interpolation == :nearest
            ix_nearest = round(Int, ix)
            for c in 1:channels
                if in_bounds(ix_nearest, iW)
                    dx[ix_nearest, c, n] += Δ[w, c, n]
                end
            end
            dgrid[1, w, n] = 0
        end
    end
end

@kernel function ∇grid_sample_kernel!(
    dx::AbstractArray{T,4},
    dgrid::AbstractArray{T,4},
    @Const(Δ::AbstractArray{T,4}),
    @Const(input::AbstractArray{T,4}),
    @Const(grid::AbstractArray{<:AbstractFloat,4}),
    ::Val{interpolation},
    ::Val{padding},
    align_corners::Bool
) where {T,interpolation,padding}

    @uniform iH, iW, channels, batch = size(input)
    w, h = @index(Global, NTuple)

    @inbounds for n in 1:batch
        x = grid[1, w, h, n]
        y = grid[2, w, h, n]
        ix, gix_mult = ∇compute_source_index(x, iW, padding, align_corners)
        iy, giy_mult = ∇compute_source_index(y, iH, padding, align_corners)

        if interpolation == :linear
            ix_nw = unsafe_trunc(Int, floor(ix))
            iy_nw = unsafe_trunc(Int, floor(iy))
            ix_ne, iy_ne = ix_nw + 1, iy_nw
            ix_sw, iy_sw = ix_nw, iy_nw + 1
            ix_se, iy_se = ix_ne, iy_sw

            nw = (ix_se - ix) * (iy_se - iy)
            ne = (ix - ix_sw) * (iy_sw - iy)
            sw = (ix_ne - ix) * (iy - iy_ne)
            se = (ix - ix_nw) * (iy - iy_nw)

            gix = zero(T)
            giy = zero(T)

            for c in 1:channels
                g_out = Δ[w, h, c, n]

                if in_bounds(iy_nw, ix_nw, iH, iW)
                    Atomix.@atomic dx[ix_nw, iy_nw, c, n] += g_out * nw
                    nw_val = input[ix_nw, iy_nw, c, n]
                    gix -= nw_val * (iy_se - iy) * g_out
                    giy -= nw_val * (ix_se - ix) * g_out
                end
                if in_bounds(iy_ne, ix_ne, iH, iW)
                    Atomix.@atomic dx[iy_ne, ix_ne, c, n] += g_out * ne
                    ne_val = input[iy_ne, ix_ne, c, n]
                    gix += ne_val * (iy_sw - iy) * g_out
                    giy -= ne_val * (ix - ix_sw) * g_out
                end
                if in_bounds(iy_sw, ix_sw, iH, iW)
                    Atomix.@atomic dx[iy_sw, ix_sw, c, n] += g_out * sw
                    sw_val = input[iy_sw, ix_sw, c, n]
                    gix -= sw_val * (iy - iy_ne) * g_out
                    giy += sw_val * (ix_ne - ix) * g_out
                end
                if in_bounds(iy_se, ix_se, iH, iW)
                    Atomix.@atomic dx[iy_se, ix_se, c, n] += g_out * se
                    se_val = input[iy_se, ix_se, c, n]
                    gix += se_val * (iy - iy_nw) * g_out
                    giy += se_val * (ix - ix_nw) * g_out
                end
            end

            dgrid[1, w, h, n] = gix_mult * gix
            dgrid[2, w, h, n] = giy_mult * giy

        elseif interpolation == :nearest
            ix_nearest = round(Int, ix)
            iy_nearest = round(Int, iy)
            for c in 1:channels
                if in_bounds(iy_nearest, ix_nearest, iH, iW)
                    Atomix.@atomic dx[ix_nearest, iy_nearest, c, n] += Δ[w, h, c, n]
                end
            end
            dgrid[1, w, h, n] = 0
            dgrid[2, w, h, n] = 0
        end
    end
end

@kernel function ∇grid_sample_kernel!(
    dx::AbstractArray{T,5},
    dgrid::AbstractArray{T,5},
    @Const(Δ::AbstractArray{T,5}),
    @Const(input::AbstractArray{T,5}),
    @Const(grid::AbstractArray{<:AbstractFloat,5}),
    ::Val{interpolation},
    ::Val{padding},
    align_corners::Bool
) where {T,interpolation,padding}

    @uniform iD, iH, iW, channels, batch = size(input)
    d, h, w = @index(Global, NTuple{3,Int})

    @inbounds for n in 1:batch
        x = grid[1, d, h, w, n]
        y = grid[2, d, h, w, n]
        z = grid[3, d, h, w, n]

        ix, gix_mult = ∇compute_source_index(x, iW, padding, align_corners)
        iy, giy_mult = ∇compute_source_index(y, iH, padding, align_corners)
        iz, giz_mult = ∇compute_source_index(z, iD, padding, align_corners)

        if interpolation == :linear
            # Get the top-north-west corner
            ix_tnw = unsafe_trunc(Int, floor(ix))
            iy_tnw = unsafe_trunc(Int, floor(iy))
            iz_tnw = unsafe_trunc(Int, floor(iz))

            # Define all corners based on tnw
            # top-north-east
            ix_tne = ix_tnw + 1
            iy_tne = iy_tnw
            iz_tne = iz_tnw

            # top-south-west
            ix_tsw = ix_tnw
            iy_tsw = iy_tnw + 1
            iz_tsw = iz_tnw

            # top-south-east
            ix_tse = ix_tnw + 1
            iy_tse = iy_tnw + 1
            iz_tse = iz_tnw

            # bottom-north-west
            ix_bnw = ix_tnw
            iy_bnw = iy_tnw
            iz_bnw = iz_tnw + 1

            # bottom-north-east
            ix_bne = ix_tnw + 1
            iy_bne = iy_tnw
            iz_bne = iz_tnw + 1

            # bottom-south-west
            ix_bsw = ix_tnw
            iy_bsw = iy_tnw + 1
            iz_bsw = iz_tnw + 1

            # bottom-south-east
            ix_bse = ix_tnw + 1
            iy_bse = iy_tnw + 1
            iz_bse = iz_tnw + 1

            # Calculate interpolation weights
            tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
            tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
            tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
            tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
            bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
            bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
            bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
            bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

            gix = zero(T)
            giy = zero(T)
            giz = zero(T)

            for c in 1:channels
                g_out = Δ[d, h, w, c, n]

                # Apply weights for all 8 corners
                if in_bounds(ix_tnw, iy_tnw, iz_tnw, iW, iH, iD)
                    Atomix.@atomic dx[ix_tnw, iy_tnw, iz_tnw, c, n] += g_out * tnw
                    tnw_val = input[ix_tnw, iy_tnw, iz_tnw, c, n]
                    gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * g_out
                    giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * g_out
                    giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * g_out
                end
                if in_bounds(ix_tne, iy_tne, iz_tne, iW, iH, iD)
                    Atomix.@atomic dx[ix_tne, iy_tne, iz_tne, c, n] += g_out * tne
                    tne_val = input[ix_tne, iy_tne, iz_tne, c, n]
                    gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * g_out
                    giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * g_out
                    giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * g_out
                end
                if in_bounds(ix_tsw, iy_tsw, iz_tsw, iW, iH, iD)
                    Atomix.@atomic dx[ix_tsw, iy_tsw, iz_tsw, c, n] += g_out * tsw
                    tsw_val = input[ix_tsw, iy_tsw, iz_tsw, c, n]
                    gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * g_out
                    giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * g_out
                    giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * g_out
                end
                if in_bounds(ix_tse, iy_tse, iz_tse, iW, iH, iD)
                    Atomix.@atomic dx[ix_tse, iy_tse, iz_tse, c, n] += g_out * tse
                    tse_val = input[ix_tse, iy_tse, iz_tse, c, n]
                    gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * g_out
                    giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * g_out
                    giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * g_out
                end
                if in_bounds(ix_bnw, iy_bnw, iz_bnw, iW, iH, iD)
                    Atomix.@atomic dx[ix_bnw, iy_bnw, iz_bnw, c, n] += g_out * bnw
                    bnw_val = input[ix_bnw, iy_bnw, iz_bnw, c, n]
                    gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * g_out
                    giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * g_out
                    giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * g_out
                end
                if in_bounds(ix_bne, iy_bne, iz_bne, iW, iH, iD)
                    Atomix.@atomic dx[ix_bne, iy_bne, iz_bne, c, n] += g_out * bne
                    bne_val = input[ix_bne, iy_bne, iz_bne, c, n]
                    gix += bne_val * (iy - iy_tsw) * (iz - iz_tsw) * g_out
                    giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * g_out
                    giz += bne_val * (ix - ix_tsw) * (iy - iy_tsw) * g_out
                end
                if in_bounds(ix_bsw, iy_bsw, iz_bsw, iW, iH, iD)
                    Atomix.@atomic dx[ix_bsw, iy_bsw, iz_bsw, c, n] += g_out * bsw
                    bsw_val = input[ix_bsw, iy_bsw, iz_bsw, c, n]
                    gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * g_out
                    giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * g_out
                    giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * g_out
                end
                if in_bounds(ix_bse, iy_bse, iz_bse, iW, iH, iD)
                    Atomix.@atomic dx[ix_bse, iy_bse, iz_bse, c, n] += g_out * bse
                    bse_val = input[ix_bse, iy_bse, iz_bse, c, n]
                    gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * g_out
                    giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * g_out
                    giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * g_out
                end
            end

            dgrid[1, d, h, w, n] = gix_mult * gix
            dgrid[2, d, h, w, n] = giy_mult * giy
            dgrid[3, d, h, w, n] = giz_mult * giz

        elseif interpolation == :nearest
            ix_nearest = round(Int, ix)
            iy_nearest = round(Int, iy)
            iz_nearest = round(Int, iz)

            for c in 1:channels
                if in_bounds(ix_nearest, iy_nearest, iz_nearest, iW, iH, iD)
                    Atomix.@atomic dx[ix_nearest, iy_nearest, iz_nearest, c, n] += Δ[d, h, w, c, n]
                end
            end
            dgrid[1, d, h, w, n] = 0
            dgrid[2, d, h, w, n] = 0
            dgrid[3, d, h, w, n] = 0
        end
    end
end
