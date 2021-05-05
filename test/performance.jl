@testset "PerformanceEvent encoding and decoding" begin
    perf = Performance(velocity_bins = 16)
    pairs = [
        (PerformanceEvent(NOTE_ON, 60), 60),
        (PerformanceEvent(NOTE_ON, 0), 0),
        (PerformanceEvent(NOTE_ON, 22), 22),
        (PerformanceEvent(NOTE_ON, 127), 127),
        (PerformanceEvent(NOTE_OFF, 72), 200),
        (PerformanceEvent(NOTE_OFF, 0), 128),
        (PerformanceEvent(NOTE_OFF, 22), 150),
        (PerformanceEvent(NOTE_OFF, 127), 255),
        (PerformanceEvent(TIME_SHIFT, 10), 265),
        (PerformanceEvent(TIME_SHIFT, 1), 256),
        (PerformanceEvent(TIME_SHIFT, 72), 327),
        (PerformanceEvent(TIME_SHIFT, 100), 355),
        (PerformanceEvent(VELOCITY, 5), 360),
        (PerformanceEvent(VELOCITY, 1), 356),
        (PerformanceEvent(VELOCITY, 16), 371)
    ]

    for (event, index) in pairs
        @test index == encodeindex(event, perf)
        @test event == decodeindex(index, perf)
    end
end
