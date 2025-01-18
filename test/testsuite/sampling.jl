function sampling_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    gradtest_fn = Backend == CPU ? gradtest : gputest
    T = Float32 # TODO test against all supported eltypes for each backend.
    atol = T == Float32 ? 1e-3 : 1e-6

    @testset "1D Grid Sampling Forward" begin
        width = 8
        channels = 3
        batch_size = 2

        @testset "1D Input validation" begin
            input = device(rand(Float32, width, channels, batch_size))
            # Valid grid
            grid = device(rand(Float32, 1, width, batch_size))
            @test_nowarn grid_sample(input, grid)

            # Invalid batch size
            invalid_grid = device(rand(Float32, 1, width, batch_size + 1))
            @test_throws AssertionError grid_sample(input, invalid_grid)
        end

        @testset "1D Interpolation modes" begin
            input = device(rand(Float32, width, channels, batch_size))
            grid = device(rand(Float32, 1, width, batch_size))
            grid = grid .* 2 .- 1  # Scale to [-1, 1]

            @test_nowarn output_linear = grid_sample(input, grid, 
                interpolation_mode="linear")
            
            @test_nowarn output_nearest = grid_sample(input, grid, 
                interpolation_mode="nearest")

            output_linear = Array(grid_sample(input, grid, interpolation_mode="linear"))
            output_nearest = Array(grid_sample(input, grid, interpolation_mode="nearest"))

            # Results should be different
            @test output_linear != output_nearest
        end

        @testset "1D Padding modes" begin
            input = device(ones(Float32, width, channels, batch_size))
            # Create grid with out-of-bounds coordinates
            grid = device(ones(Float32, 1, width, batch_size) .* 2)  # All coordinates > 1

            output_zeros = Array(grid_sample(input, grid, padding_mode="zeros"))
            @test output_zeros == zeros(Float32, width, channels, batch_size)

            output_border = Array(grid_sample(input, grid, padding_mode="border"))
            @test output_border == ones(Float32, width, channels, batch_size)
        end

        @testset "1D Alignment modes" begin
            input = device(Float32[0.04298234;0.4032818;0.062031806;0.23488349;0.12891841;0.20078379;0.079730034;0.20719224;;;])
            grid = device(reshape(collect(range(-1f0, 1f0, length=width)), (1,width,1)))
            output_aligned   = Array(grid_sample(input, grid, align_corners=true))
            output_unaligned = Array(grid_sample(input, grid, align_corners=false))
            @test output_aligned != output_unaligned
        end

        @testset "1D Absolute Correctness" begin
            input = device(reshape(collect(1.:3.), (3,1,1)))
            grid  = device(reshape(collect(-1.:0.5:1.), (1,5,1)))

            @testset "Linear" begin
                output = Array(grid_sample(input, grid; interpolation_mode="linear", align_corners=true, padding_mode="border"))
                @test output == reshape([1, 1.5, 2, 2.5, 3], (5,1,1))
                output = Array(grid_sample(input, grid; interpolation_mode="linear", align_corners=true, padding_mode="zeros"))
                @test output == reshape([1, 1.5, 2, 2.5, 3], (5,1,1))

                output = Array(grid_sample(input, grid; interpolation_mode="linear", align_corners=false, padding_mode="border"))
                @test output == reshape([1, 1.25, 2, 2.75, 3], (5,1,1))
                output = Array(grid_sample(input, grid; interpolation_mode="linear", align_corners=false, padding_mode="zeros"))
                @test output == reshape([0.5, 1.25, 2, 2.75, 1.5], (5,1,1))
            end

            @testset "Nearest" begin
                output = Array(grid_sample(input, grid; interpolation_mode="nearest", align_corners=true, padding_mode="border"))
                @test output == reshape([1, 2, 2, 2, 3], (5,1,1))
                output = Array(grid_sample(input, grid; interpolation_mode="nearest", align_corners=true, padding_mode="zeros"))
                @test output == reshape([1, 2, 2, 2, 3], (5,1,1))

                output = Array(grid_sample(input, grid; interpolation_mode="nearest", align_corners=false, padding_mode="border"))
                @test output == reshape([1, 1, 2, 3, 3], (5,1,1))
                output = Array(grid_sample(input, grid; interpolation_mode="nearest", align_corners=false, padding_mode="zeros"))
                @test output == reshape([0, 1, 2, 3, 0], (5,1,1))
            end
        end
    end

    @testset "1D Grid Sampling Backward" begin
        @testset "dimensions" begin
            width = 8
            channels = 2
            batch_size = 2

            Δ     = device(ones(Float32, width, channels, batch_size))
            input = device(ones(Float32, width, channels, batch_size))
            grid  = device(ones(Float32, 1, width, batch_size))

            ∇input, ∇grid = ∇grid_sample(Δ, input, grid)
            ∇input, ∇grid = Array(∇input), Array(∇grid)

            @test size(∇input) == size(input)
            @test size(∇grid) == size(grid)
        end
        
        @testset "same size sampling" begin
            channels = 1
            batch_size = 1

            @testset "interpolation=linear" begin
                @testset "align_corners=true" begin
                    for width in (3,4,5)
                        Δ = device(ones(Float32, width, channels, batch_size))
                        input = device(ones(Float32, width, channels, batch_size))
                        grid = device(zeros(Float32, 1, width, batch_size))
                        grid[1, :, 1] .= device(collect(range(-1, 1, length=width)))
                        
                        ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:zeros, align_corners=true)
                        ∇input, ∇grid = Array(∇input), Array(∇grid)
                        @test ∇input == ones(Float32, width, channels, batch_size)
                        ∇grid_expected = zeros(Float32, 1, width, batch_size)
                        ∇grid_expected[1,end,1] = -0.5 * (width - 1)
                        @test ∇grid == ∇grid_expected

                        ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:border, align_corners=true)
                        ∇input, ∇grid = Array(∇input), Array(∇grid)
                        @test ∇input == ones(Float32, width, channels, batch_size)
                        @test ∇grid == zeros(Float32, 1, width, batch_size)
                    end
                end
                @testset "align_corners=false" begin
                    width = 3
                    Δ = device(ones(Float32, width, channels, batch_size))
                    input = device(ones(Float32, width, channels, batch_size))
                    grid = device(zeros(Float32, 1, width, batch_size))
                    grid[1, :, 1] .= device(collect(range(-1, 1, length=width)))
                    
                    ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:zeros, align_corners=false)
                    ∇input, ∇grid = Array(∇input), Array(∇grid)
                    @test ∇input ≈ [0.5, 1, 0.5][:,:,:]
                    @test ∇grid == reshape([1.5, fill(0, width-2)..., -1.5], 1, width, batch_size)

                    ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:border, align_corners=false)
                    ∇input, ∇grid = Array(∇input), Array(∇grid)
                    @test ∇input ≈ ones(Float32, width, channels, batch_size)
                    @test ∇grid == zeros(Float32, 1, width, batch_size)
                    
                    width = 4
                    Δ = device(ones(Float32, width, channels, batch_size))
                    input = device(ones(Float32, width, channels, batch_size))
                    grid = device(zeros(Float32, 1, width, batch_size))
                    grid[1, :, 1] .= device(collect(range(-1, 1, length=width)))
                    
                    ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:zeros, align_corners=false)
                    ∇input, ∇grid = Array(∇input), Array(∇grid)
                    @test ∇input ≈ Float32[2/3, 5/6, 5/6, 2/3][:,:,:]
                    @test ∇grid == Float32[2 0 0 -2;;;]

                    ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:border, align_corners=false)
                    ∇input, ∇grid = Array(∇input), Array(∇grid)
                    @test ∇input ≈ Float32[7/6, 5/6, 5/6, 7/6][:,:,:]
                    @test ∇grid == zeros(Float32, 1, width, batch_size)

                    width = 5
                    Δ = device(ones(Float32, width, channels, batch_size))
                    input = device(ones(Float32, width, channels, batch_size))
                    grid = device(zeros(Float32, 1, width, batch_size))
                    grid[1, :, 1] .= device(collect(range(-1, 1, length=width)))
                    
                    ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:zeros, align_corners=false)
                    ∇input, ∇grid = Array(∇input), Array(∇grid)
                    @test ∇input ≈ [0.75, 0.75, 1, 0.75, 0.75][:,:,:]
                    @test ∇grid == reshape([2.5, fill(0, width-2)..., -2.5], 1, width, batch_size)

                    ∇input, ∇grid = ∇grid_sample(Δ, input, grid; padding_mode=:border, align_corners=false)
                    ∇input, ∇grid = Array(∇input), Array(∇grid)
                    @test ∇input ≈ [1.25, 0.75, 1, 0.75, 1.25][:,:,:]
                    @test ∇grid == zeros(Float32, 1, width, batch_size)
                end
            end
        end
    end
        
    @testset "2D Grid Sampling Forward" begin
        @testset "5x5 input and grid" begin
            # Prepare input: 5x5 tensor of ones
            input = device(ones(Float32, 5, 5, 1, 1))
            
            # Create grid: x and y coordinates ranging from -1 to 1
            v = collect(range(-1f0, 1f0, length=5))
            xgrid = repeat(v', 5)
            ygrid = repeat(v, 1, 5)
            grid = reshape(device(Float32.(stack((xgrid, ygrid), dims=1))), 2, 5, 5, 1)
        
            # Test cases with different parameters
            test_cases = [
                # align_corners=true, interpolation_mode=nearest, padding_mode=zeros
                (true, :nearest, :zeros, ones(Float32, 5, 5, 1, 1)),
                # align_corners=true, interpolation_mode=nearest, padding_mode=border
                (true, :nearest, :border, ones(Float32, 5, 5, 1, 1)),
                # align_corners=true, interpolation_mode=linear, padding_mode=zeros
                (true, :linear, :zeros, ones(Float32, 5, 5, 1, 1)),
                # align_corners=true, interpolation_mode=linear, padding_mode=border
                (true, :linear, :border, ones(Float32, 5, 5, 1, 1)),
                # align_corners=false, interpolation_mode=nearest, padding_mode=zeros
                (false, :nearest, :zeros, 
                Float32[0.0  0.0  0.0  0.0  0.0
                        0.0  1.0  1.0  1.0  0.0
                        0.0  1.0  1.0  1.0  0.0
                        0.0  1.0  1.0  1.0  0.0
                        0.0  0.0  0.0  0.0  0.0][:,:,:,:]),
                # align_corners=false, interpolation_mode=nearest, padding_mode=border
                (false, :nearest, :border, ones(Float32, 5, 5, 1, 1)),
                # align_corners=false, interpolation_mode=linear, padding_mode=zeros
                (false, :linear, :zeros, Float32[
                    0.25 0.5 0.5 0.5 0.25;
                    0.5  1.0 1.0 1.0 0.5;
                    0.5  1.0 1.0 1.0 0.5;
                    0.5  1.0 1.0 1.0 0.5;
                    0.25 0.5 0.5 0.5 0.25
                ][:,:,:,1:1]),
                # align_corners=false, interpolation_mode=linear, padding_mode=border
                (false, :linear, :border, ones(Float32, 5, 5, 1, 1))
            ]
        
            for (align_corners, interpolation_mode, padding_mode, expected) in test_cases
                output = Array(grid_sample(device(input), device(grid); 
                    interpolation_mode=interpolation_mode, 
                    padding_mode=padding_mode, 
                    align_corners=align_corners))
                @test output ≈ expected rtol=0.0001f0
            end
        end

        @testset "2D out-of-bounds for different paddings" begin
            x = device(ones(Float64, (2, 2, 1, 1)))
            grid = device(Array{Float64}(undef, 2, 3, 2, 1))
            grid[:, 1, 1, 1] .= (-3, -1)
            grid[:, 2, 1, 1] .= (0, -1)
            grid[:, 3, 1, 1] .= (3, -1)
            grid[:, 1, 2, 1] .= (-1, 3)
            grid[:, 2, 2, 1] .= (0, 1)
            grid[:, 3, 2, 1] .= (1, 3)
        
            # With 0-padding, out-of-bound values are will contribute nothing to
            # the output values, because they are too far from any bound.
            y = grid_sample(x, grid; padding_mode=:zeros)
            y_true = device(reshape(Float64[[0, 1, 0] [0, 1, 0]], size(y)))
            @test y ≈ y_true
        
            # With border-padding, out-of-bound values simly become border values
            # and the result should be all ones.
            y = grid_sample(x, grid; padding_mode=:border)
            y_true = device(ones(Float64, size(y)))
            @test y ≈ y_true
        end
    end
    
    @testset "2D Grid Sampling Backward" begin
        @testset "5x5 input and grid" begin
            # Prepare input: 5x5 tensor of ones
            input = device(ones(Float32, 5, 5, 1, 1))
            
            # Create grid: x and y coordinates ranging from -1 to 1
            v = collect(range(-1f0, 1f0, length=5))
            xgrid = repeat(v', 5)
            ygrid = repeat(v, 1, 5)
            grid  = device(reshape(Float32.(stack((xgrid, ygrid), dims=1)), 2, 5, 5, 1))
        
            # Test cases with different parameters
            test_cases = [
                # align_corners=true, interpolation_mode=nearest, padding_mode=zeros
                (true, :nearest, :zeros, 
                ones(Float32, 5, 5, 1, 1),  # dx
                zeros(Float32, 2, 5, 5, 1)), # dgrid
                # align_corners=true, interpolation_mode=nearest, padding_mode=border
                (true, :nearest, :border,
                ones(Float32, 5, 5, 1, 1),
                zeros(Float32, 2, 5, 5, 1)),
                # align_corners=true, interpolation_mode=linear, padding_mode=zeros
                (true, :linear, :zeros,
                ones(Float32, 5, 5, 1, 1),
                Float32.(reshape(stack(
                    (
                    [0 0 0 0 -2; 0 0 0 0 -2; 0 0 0 0 -2; 0 0 0 0 -2; 0 0 0 0 -2],  # dx
                    [0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; -2 -2 -2 -2 -2]   # dy
                    ); dims=1), 2, 5, 5, 1)),
                ),
                # align_corners=true, interpolation_mode=linear, padding_mode=border
                (true, :linear, :border,
                ones(Float32, 5, 5, 1, 1),
                zeros(Float32, 2, 5, 5, 1)),
                # align_corners=false, interpolation_mode=nearest, padding_mode=zeros
                (false, :nearest, :zeros,
                Float32[0.0  0.0  0.0  0.0  0.0
                        0.0  1.0  1.0  1.0  0.0
                        0.0  1.0  1.0  1.0  0.0
                        0.0  1.0  1.0  1.0  0.0
                        0.0  0.0  0.0  0.0  0.0][:,:,:,:],
                zeros(Float32, 2, 5, 5, 1)),
                # align_corners=false, interpolation_mode=nearest, padding_mode=border
                (false, :nearest, :border,
                ones(Float32, 5, 5, 1, 1),
                zeros(Float32, 2, 5, 5, 1)),
                # align_corners=false, interpolation_mode=linear, padding_mode=zeros
                (false, :linear, :zeros,
                reshape(Float32[
                    0.5625 0.5625 0.7500 0.5625 0.5625;
                    0.5625 0.5625 0.7500 0.5625 0.5625;
                    0.7500 0.7500 1.0000 0.7500 0.7500;
                    0.5625 0.5625 0.7500 0.5625 0.5625;
                    0.5625 0.5625 0.7500 0.5625 0.5625
                ], 5, 5, 1, 1),
                Float32.(reshape(stack(
                    (
                    [1.25 0.0 0.0 0.0 -1.25; 2.5 0.0 0.0 0.0 -2.5; 2.5 0.0 0.0 0.0 -2.5; 2.5 0.0 0.0 0.0 -2.5; 1.25 0.0 0.0 0.0 -1.25], # dx
                    [1.25 2.5 2.5 2.5 1.25; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0; -1.25 -2.5 -2.5 -2.5 -1.25] # dy
                    ); dims=1), 2, 5, 5, 1)),
                ),
                # align_corners=false, interpolation_mode=linear, padding_mode=border
                (false, :linear, :border,
                reshape(Float32[
                    1.5625 0.9375 1.2500 0.9375 1.5625;
                    0.9375 0.5625 0.7500 0.5625 0.9375;
                    1.2500 0.7500 1.0000 0.7500 1.2500;
                    0.9375 0.5625 0.7500 0.5625 0.9375;
                    1.5625 0.9375 1.2500 0.9375 1.5625
                ], 5, 5, 1, 1),
                zeros(Float32, 2, 5, 5, 1))
            ]
        
            for (align_corners, interpolation_mode, padding_mode, expected_dx, expected_dgrid) in test_cases
                # Compute forward pass
                output = grid_sample(input, grid; 
                    interpolation_mode=interpolation_mode, 
                    padding_mode=padding_mode, 
                    align_corners=align_corners)
                
                Δ = device(ones(Float32, size(output)...))
                # Compute backward pass
                ∇input, ∇grid = ∇grid_sample(Δ, input, grid;
                    interpolation_mode=interpolation_mode, 
                    padding_mode=padding_mode, 
                    align_corners=align_corners)
                ∇input, ∇grid = Array(∇input), Array(∇grid)
                
                @test ∇input ≈ expected_dx rtol=0.0001f0
                @test ∇grid ≈ expected_dgrid rtol=0.0001f0
            end
        end

        @testset "tests against forwarddiff" begin
            x = device(ones(Float64, (2, 2, 1, 1)))
            grid = Array{Float64}(undef, 2, 2, 2, 1)
            grid[:, 1, 1, 1] .= (-1, -1)
            grid[:, 2, 1, 1] .= (1, -1)
            grid[:, 1, 2, 1] .= (-1, 1)
            grid[:, 2, 2, 1] .= (1, 1)
            grid = device(grid)
        
            ∇grid_true = Array{Float64}(undef, size(grid))
            ∇grid_true[:, :, 1, 1] = [[0.0, 0.0] [-0.5, 0.0]]
            ∇grid_true[:, :, 2, 1] = [[0.0, -0.5] [-0.5, -0.5]]
            ∇grid_true = device(∇grid_true)
        
            padding_mode = :zeros
            sampled = grid_sample(x, grid; padding_mode=padding_mode)
            @test sampled ≈ x
            @test eltype(sampled) == Float64
            external_grad = device(ones(size(sampled)))
            ∇input, ∇grid = ∇grid_sample(external_grad, x, grid; padding_mode=padding_mode)
            @test ∇input ≈ x
            @test ∇grid ≈ ∇grid_true
            @test eltype(∇input) == Float64
            @test eltype(∇grid) == Float64
        
            padding_mode = :border
            fill!(∇grid_true, 0.0)
            sampled = grid_sample(x, grid; padding_mode=padding_mode)
            @test sampled ≈ x
            @test eltype(sampled) == Float64
            external_grad = device(ones(size(sampled)))
            ∇input, ∇grid = ∇grid_sample(external_grad, x, grid; padding_mode=padding_mode)
            @test ∇input ≈ x
            @test ∇grid ≈ ∇grid_true
            @test eltype(∇input) == Float64
            @test eltype(∇grid) == Float64
            
            if Backend == CPU
                gradtest(grid_sample, x, grid; fkwargs=(align_corners=true, padding_mode=padding_mode, interpolation_mode=:linear))
            else
                gradtest_fn(grid_sample, x, grid; align_corners=true, padding_mode=padding_mode, interpolation_mode=:linear)
            end

            @show "starting long test"
            n = 13
            x = Float32[0.669413     0.377877  0.758927  0.0800772  0.770714    0.538992   0.0541975  0.00985956
            0.000746131  0.194609  0.19077   0.374846   0.793122    0.656515   0.186811   0.104596
            0.301428     0.868005  0.375169  0.607452   0.549318    0.0378221  0.798738   0.13069
            0.261917     0.794659  0.454242  0.638406   0.00739515  0.244023   0.31287    0.737728
            0.109388     0.236433  0.307977  0.349825   0.693461    0.361304   0.754347   0.023451
            0.195654     0.173322  0.201782  0.910598   0.839297    0.136212   0.189656   0.185308
            0.464608     0.43914   0.377606  0.283426   0.762021    0.789998   0.643871   0.489429
            0.578882     0.370316  0.374061  0.520113   0.887864    0.444599   0.908073   0.784409][:,:,:,:]
            v = collect(range(-1.1, 1.1, length=n))
            xgrid = repeat(v', n)
            ygrid = repeat(v, 1, n)
            grid  = Float32.(stack((xgrid, ygrid), dims=1))[:,:,:,:]
            
            x = device(x)
            grid = device(grid)

            # test correctness of grad wrt input
            input_fn = (z; kwargs...) -> grid_sample(z, grid; kwargs...)

            for align_corners in (true, false)
                for padding_mode in (:zeros, :border)
                    for interpolation_mode in (:linear, :nearest)
                        @testset "$align_corners, $padding_mode, $interpolation_mode" begin
                            if Backend == CPU
                                gradtest(input_fn, x; fkwargs=(align_corners=align_corners, padding_mode=padding_mode, interpolation_mode=interpolation_mode), atol=1e-3)
                            else
                                gradtest_fn(input_fn, x; align_corners=align_corners, padding_mode=padding_mode, interpolation_mode=interpolation_mode)
                            end
                        end
                    end
                end
            end
        end
    end

    
    
end