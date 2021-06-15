export notes2midi

using MIDI: SetTempoEvent, TimeSignatureEvent

"""     notes2midi(notes::Notes)
Return a `MIDIFile` from the given `Notes`.
"""
function notes2midi(
        notes::Notes;
        qpm = 120,
        timesig::TimeSignatureEvent = TimeSignatureEvent(0, 4, 4, 24, 8))
    metatrack = MIDITrack()
    events = [
        SetTempoEvent(0, Int(6e7 / qpm)),
        timesig
    ]
    addevents!(metatrack, [0, 0], events) # Add the meta events
    notestrack = MIDITrack()
    addnotes!(notestrack, notes)
    midi = MIDIFile(1, notes.tpq, [metatrack, notestrack])
end
