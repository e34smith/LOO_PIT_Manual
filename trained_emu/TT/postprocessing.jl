function postprocessing(input, Cl, emulator)
    return Cl .* exp(input[1]-3)
end
