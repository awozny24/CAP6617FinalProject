% Gets the audio files and extracts STRAIGHT features from them

clear all;
path = 'M:\Documents\MATLAB\voice-conversion\vcc2018_training\VCC2T';
savePath = 'M:\Documents\MATLAB\voice-data\';
for g = ['F', 'M']
    for i = 1:2
        for num = 1:81
            fprintf("%i\n", num);
            if num < 10
                last = append("0", string(num));
            else
                last = string(num);
            end
            file = append(path, g, string(i),'\',  "100", last,'.wav');
            [x, fs] = audioread(file); 
            f0raw = MulticueF0v14(x,fs); 
            ap = exstraightAPind(x,fs,f0raw);
            n3sgram=exstraightspec(x,f0raw,fs);
            savedFile = append(savePath, "Trgt", g, string(i), '_', last);
            save(savedFile, 'f0raw', 'ap', 'n3sgram');
            clear f0raw ap n3sgram;
        end
    end
end

