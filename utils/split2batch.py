def split2batch(parts, duration):

    intervals = []

    part_duration = int(duration / parts)

    if part_duration == 0:
        intervals.append((0, duration))
        return intervals
    
    for i in range(part_duration+1):
        start = i * parts
        stop = (i + 1) * parts

        if stop >= duration:
            stop = duration
            intervals.append((start, stop))
            return intervals
        
        intervals.append((start, stop))