export notes2midi

function notes2midi(notes::Notes)
    metatrack = MIDITrack()
    # TODO:
    #   Find a better way to do this
    events = [
        MetaEvent(0, 0x51, UInt8[0x07, 0xa1, 0x20])        # 120 QPM
        MetaEvent(0, 0x58, UInt8[0x04, 0x02, 0x18, 0x08])  # 4/4 Time Signature
    ]
    addevents!(metatrack, [0, 0], events) # Add defaults
    notestrack = MIDITrack()
    addnotes!(notestrack, notes)
    midi = MIDIFile(1, notes.tpq, [metatrack, notestrack])
end
