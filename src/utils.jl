export notes2midi

"""     notes2midi(notes::Notes)
Return a `MIDIFile` from the given `Notes`.
"""
function notes2midi(notes::Notes; qpm = 120)
    metatrack = MIDITrack()
    # TODO:
    #   Find a better way to set time signature
    qpm_bytes = reinterpret(UInt8, [UInt32(6e7 / qpm)])
    events = [
        MetaEvent(0, 0x51, reverse(qpm_bytes[1:3])),
        MetaEvent(0, 0x58, UInt8[0x04, 0x02, 0x18, 0x08])  # 4/4 Time Signature
    ]
    addevents!(metatrack, [0, 0], events) # Add defaults
    notestrack = MIDITrack()
    addnotes!(notestrack, notes)
    midi = MIDIFile(1, notes.tpq, [metatrack, notestrack])
end
