function compute_streamlines(scene_name, out_scene_name, sim_start, sim_end, start_frame, end_frame)
    %%--
    % set the python environment before running
    % Undefined variable "py" or function "py.command" when you type a py. command.
    % pyversion D:\Python\Python35\python.exe
    %%--
    
    % --- default path variables
    in_manta_base_folder                    = '../manta/sbfs_scenes/data/';    % uni data
    in_base_filename_vel_grid               = 'flipVel_%04d.uni';
    in_base_filename_lvst_grid              = 'flipLevelSet_%04d.uni';
    out_mat_base_folder                     = 'data/';

    % --- filenames
    out_base_filename_streamline            = 'flipStreamline_%04d.txt';
    out_base_filename_resampled_streamline  = 'flipStreamline_%04d_resampled.txt';
    out_base_filename_sketch_grid           = 'flipSketchGrid_%04d.bin';

    % --- scene setting
    grid_dim                                = 3;
    grid_res                                = 128;

    num_streamline_vertices                 = 256;       % number of streamline vertices
    num_seeds                             = 4096;      % number of seed points (4096 = 16*16*16)
    max_streamline_keep                   = 150;       % maximum number of streamline needed
    % num_seeds                               = 8192;      % number of seed points (4096 = 16*16*32)
    % max_streamline_keep                     = 200;       % maximum number of streamline needed
    % num_seeds                             = 16384;     % number of seed points (4096 = 16*32*32)
    % max_streamline_keep                   = 250;       % maximum number of streamline needed
    % num_seeds                             = 32768;     % number of seed points (4096 = 32*32*32)
    % max_streamline_keep                   = 300;       % maximum number of streamline needed
    
    min_streamline_length                   = 5;         % minimum streamline length needed (world coordinate)

    save_figures                            = false;
    save_sketches                           = true;
    verbose                                 = false;

    tic             % measure the computing time needed
    % ticBytes(gcp);  % measure how much data is transferred to and from the workers in the parallel pool
    % --- loop over all simulation scenes
    % increase memory size to prevent out of memory error
    delete(gcp('nocreate')) % terminate the existing session
    parpool('local',6) % creates a pool with the specified number of workers
    for sim_num = sim_start:sim_end
        in_folder   = strcat(in_manta_base_folder, scene_name, num2str(sim_num));
        out_folder  = strcat(out_mat_base_folder, out_scene_name, num2str(sim_num));
        if(~exist(in_folder, 'dir')),   error('input folder %s not exists\n', in_folder); end
        if(~exist(out_folder, 'dir')),  mkdir(out_folder);    end
        fprintf('start working on folder %s\n', in_folder);

        % --- loop over all simulation frames
        % ref: https://www.mathworks.com/help/distcomp/troubleshoot-variables-in-parfor-loops.html
        parfor frame = start_frame : end_frame
            %- I/O filenames
            in_filename_vel_grid                = sprintf(in_base_filename_vel_grid, frame);
            in_filename_lvst_grid               = sprintf(in_base_filename_lvst_grid, frame);
            out_filename_streamline             = sprintf(out_base_filename_streamline, frame);
            out_filename_resampled_streamline   = sprintf(out_base_filename_resampled_streamline, frame);
            out_filename_sketch_grid            = sprintf(out_base_filename_sketch_grid, frame);
            fprintf('\t start working on %s and %s\n', in_filename_vel_grid, in_filename_lvst_grid);

            %- load velocity vector field and levelset scale field
            if(~exist(strcat(in_folder,'/',in_filename_vel_grid), 'file') || ~exist(strcat(in_folder,'/',in_filename_lvst_grid), 'file'))
                error('cannot open %s\n', strcat(in_folder,'/',in_filename_vel_grid));
            end
            %-- read uni format data (call the python function in uniio.py, make sure it is visible under the directory)
            uni_vel_data        = py.uniio.readUni(strcat(in_folder,'/',in_filename_vel_grid));
            uni_lvst_data       = py.uniio.readUni(strcat(in_folder,'/',in_filename_lvst_grid));
            %--- 1*2 tuple = (header data, content data)
            uni_vel_content     = uni_vel_data(2);
            uni_lvst_content    = uni_lvst_data(2);
            %--- convert data format to matlab (default numeric type for MATLAB is double)
            py_vel_content      = py.numpy.asarray(uni_vel_content);
            py_lvst_content     = py.numpy.asarray(uni_lvst_content);
            vel_field           = double(py.array.array('f', py.numpy.nditer(py_vel_content)));
            lvst_field          = double(py.array.array('f', py.numpy.nditer(py_lvst_content)));
            %--- vel_field(:, x+1, y+1, z+1) =  manta_grid(x,y,z) because matlab is 1-based indexing
            vel_field           = reshape(vel_field, [grid_dim, grid_res, grid_res, grid_res]);
            vel_U               = squeeze(vel_field(1,:, :, :)); % remove singleton dimensions
            vel_V               = squeeze(vel_field(2,:, :, :));
            vel_W               = squeeze(vel_field(3,:, :, :));
            lvst_field          = reshape(lvst_field, [grid_res, grid_res, grid_res]);

            %- smooth the velocity field?
            % vel_U = smooth3(vel_U, 'box', [5 5 5]);
            % vel_V = smooth3(vel_V, 'box', [5 5 5]);
            % vel_W = smooth3(vel_W, 'box', [5 5 5]);

            %- clean the velocity field, using levelset as mask (fluid region)
            fluid_mask = (lvst_field <= 1.0);
            vel_U = vel_U.*fluid_mask;
            vel_V = vel_V.*fluid_mask;
            vel_W = vel_W.*fluid_mask;

            %- permute arrays to match the meshgrid function for the coordinate of the vector field,
            % which use the MATLAB indexing system: horizontal is x-axis(second argu) and vertical is y-axis(first argu)
            % thus the coordinate for grid(x,y,z) is (X(y,x), Y(y,x), Z(y,x))
            % to make it compatible, permute to grid(y,x,z)
            vel_U           = permute(vel_U, [2 1 3]);
            vel_V           = permute(vel_V, [2 1 3]);
            vel_W           = permute(vel_W, [2 1 3]);

            %- Compute 3-D streamline data
            figure('Visible', 'off' );
            [M,N,P]         = size(vel_U);
            [X,Y,Z]         = meshgrid(0:N-1,0:M-1,0:P-1);
            %-- generate random seeding point
            startx              = 0 + (grid_res - 1 ) * rand(num_seeds,1);
            starty              = 0 + (grid_res - 1 ) * rand(num_seeds,1);
            startz              = 0 + (grid_res - 1 ) * rand(num_seeds,1);
            %-- remove start point outside the mask?
            %         invalid_seeds_id  = [];
            %         for i = 1:num_seeds
            %             if not(fluid_mask(int32(startx(i)) + 1, int32(starty(i)) + 1, int32(startz(i))+ 1))
            %                 invalid_seeds_id = [invalid_seeds_id, i];
            %             end
            %         end
            %         startx(invalid_seeds_id) = [];
            %         starty(invalid_seeds_id) = [];
            %         startz(invalid_seeds_id) = [];
            %         [num_seeds, temp]        = size(startx);
            if(verbose), fprintf("\t\t streamline seeds = %d\n", num_seeds); end
            if(num_seeds == 0), error('\t Error: num sample is zero!\n'); end
            %--- generate streamlines from vector data U, V, W.
            % X, Y, and Z, which define the coordinates for U, V, and W
            % options: [stepsize, max_number_vertices]
            XYZ             = stream3(X,Y,Z,vel_U,vel_V,vel_W,startx,starty,startz);
            line_handle     = streamline(XYZ);
            if(verbose), fprintf('\t\t streamline generated\n'); end
            %--- sort the streamlines by length
            line_length         = zeros(1, num_seeds, 'single');
            line_XData_sorted   = cell(1, num_seeds);
            line_YData_sorted   = cell(1, num_seeds);
            line_ZData_sorted   = cell(1, num_seeds);
            %---- compute streamline length
            for i = 1:num_seeds
                Data_matrix = [line_handle(i).XData; line_handle(i).YData; line_handle(i).ZData]';
                Data_diff = diff(Data_matrix, 1, 1);   % first order difference between the rows
                line_length(i) = sum(sqrt(sum(Data_diff.*Data_diff, 2)));
            end
            [line_length_sorted, line_length_sorted_index] = sort(line_length, 'descend');
            %---- keep the longest lines
            num_min_line        = sum(line_length_sorted(:) > min_streamline_length);
            num_streamlines = min([max_streamline_keep, num_min_line]);
            for i = 1:num_streamlines
                line_XData_sorted{1,i}      = line_handle(line_length_sorted_index(i)).XData;
                line_YData_sorted{1,i}      = line_handle(line_length_sorted_index(i)).YData;
                line_ZData_sorted{1,i}      = line_handle(line_length_sorted_index(i)).ZData;
            end
            fprintf('\t\t number streamline = %d\n', num_streamlines);

            %- write to file: each row contain points of the line
            fileOutID = fopen(strcat(out_folder,'/',out_filename_streamline),'w');
            if(fileOutID == -1), error('\t\t cannot open %s\n', out_filename_streamline); end
            for i = 1:num_streamlines
                [~, num_points] = size(line_XData_sorted{1, i});
                for j = 1: num_points
                    fprintf(fileOutID,'%f %f %f ', line_XData_sorted{1, i}(j),line_YData_sorted{1, i}(j),line_ZData_sorted{1, i}(j));
                end
                fprintf(fileOutID,'\n');
            end
            fclose(fileOutID);

            %- resample each line to same length
            fileOutID                               = fopen(strcat(out_folder,'/',out_filename_resampled_streamline),'w');
            if(fileOutID == -1), error('\t\t cannot open %s\n', out_filename_resampled_streamline); end
            line_XData_resampled                    = cell(1,num_streamlines);
            line_YData_resampled                    = cell(1,num_streamlines);
            line_ZData_resampled                    = cell(1,num_streamlines);
            for i = 1:num_streamlines
                [~, num_points] = size(line_XData_sorted{1, i});
                for j = 1: num_streamline_vertices
                    jj = (j-1)/(num_streamline_vertices - 1)*(num_points - 1) + 1;
                    jj_prev = floor(jj);
                    jj_next = ceil(jj);
                    if(abs(jj_next - jj_prev) < 1e-8)
                        line_XData_resampled{1, i}(j) = line_XData_sorted{1, i}(jj_prev);
                        line_YData_resampled{1, i}(j) = line_YData_sorted{1, i}(jj_prev);
                        line_ZData_resampled{1, i}(j) = line_ZData_sorted{1, i}(jj_prev);
                    else
                        line_XData_resampled{1, i}(j) = line_XData_sorted{1, i}(jj_prev) + ...
                            (jj-jj_prev)/(jj_next-jj_prev)*(line_XData_sorted{1, i}(jj_next)-line_XData_sorted{1, i}(jj_prev));
                        line_YData_resampled{1, i}(j) = line_YData_sorted{1, i}(jj_prev) + ...
                            (jj-jj_prev)/(jj_next-jj_prev)*(line_YData_sorted{i}(jj_next)-line_YData_sorted{1, i}(jj_prev));
                        line_ZData_resampled{1, i}(j) = line_ZData_sorted{1, i}(jj_prev) + ...
                            (jj-jj_prev)/(jj_next-jj_prev)*(line_ZData_sorted{1, i}(jj_next)-line_ZData_sorted{1, i}(jj_prev));
                    end
                    if(isnan(line_XData_resampled{1, i}(j)) || isinf(line_XData_resampled{1, i}(j)) || ...
                            isnan(line_YData_resampled{1, i}(j)) || isinf(line_YData_resampled{1, i}(j)) || ...
                            isnan(line_ZData_resampled{1, i}(j)) || isinf(line_ZData_resampled{1, i}(j)))
                        error('\t\t ERROR! contain NAN/inf values\n');
                    end
                    fprintf(fileOutID,'%f %f %f ', line_XData_resampled{1, i}(j),line_YData_resampled{1, i}(j),line_ZData_resampled{1, i}(j));
                end
                fprintf(fileOutID,'\n');
            end
            fclose(fileOutID);
            if(verbose), fprintf('\t\t streamline saved successfully\n'); end

            %- voxelize streamlines to the grid
            if(save_sketches)
                fileOutID  = fopen(strcat(out_folder,'/',out_filename_sketch_grid),'w');
                if(fileOutID == -1), error('\t\t cannot open %s\n', out_filename_sketch_grid); end
                sketch_grid = zeros(grid_res, grid_res, grid_res);
                for i = 1:num_streamlines
                    [~, num_points] = size(line_XData_resampled{1, i});
                    for j = 1: num_points
                        p_x = line_XData_resampled{1, i}(j);
                        p_y = line_YData_resampled{1, i}(j);
                        p_z = line_ZData_resampled{1, i}(j);
                        sketch_grid(int32(p_x)+1, int32(p_y)+1, int32(p_z)+1) = 1;
                    end
                end
                % anti-aliasing: smooth
                % sketch_gid_smoothed = smooth3(sketch_grid, 'box', [3 3 3]);
                sketch_gid_smoothed = sketch_grid;
                % write binary data in the order(low dim to high dim): sketch_grid(x,y,z): x --> y --> z
                fwrite(fileOutID, sketch_gid_smoothed, 'float');
                fclose(fileOutID);
                if(verbose), fprintf('\t\t sketch data saved successfully\n'); end
            end

            % -- save figures
            if(not(save_figures)), continue; end
            figure('Name','Original Data','Visible', 'off' );
            for i = 1:num_seeds
                plot3(line_handle(i).XData, line_handle(i).YData, line_handle(i).ZData);
                axis([1 grid_res 1 grid_res 1 grid_res]);
                xlabel('X', 'Color', 'red');  ylabel('Y', 'Color', 'green');  zlabel('Z', 'Color', 'blue');
                camup([0 1 0]);
                hold on;
            end
            print('-dpng', sprintf('%s.png', strcat(out_folder,'/',out_filename_streamline)));
            %         clear uni_vel_data uni_lvst_data uni_vel_content uni_lvst_content py_vel_content py_lvst_content ...
            %             vel_field lvst_field vel_field vel_U vel_V vel_W lvst_field fluid_mask startx starty startz ...
            %             XYZ line_handle line_length line_XData_sorted line_YData_sorted line_ZData_sorted
        end
    end
    % tocBytes(gcp)
    toc
end










