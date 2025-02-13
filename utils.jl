module utils

export create_folder


function create_folder(folder_path::String)
    # Check if the folder exists
    if !isdir(folder_path)
        # Create the folder if it does not exist
        mkpath(folder_path)
        println("Folder created: $folder_path")
    end
end



end
