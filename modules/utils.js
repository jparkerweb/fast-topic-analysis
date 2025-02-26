// -------------------------------------
// -- return passed value as boolean --
// -------------------------------------
export function toBoolean(value = 'false') {
    // If value is already a boolean, return it
    if (typeof value === 'boolean') {
        return value;
    }
    
    // If value is not a string, convert it to string
    if (typeof value !== 'string') {
        value = String(value);
    }
    
    const normalizedStr = value.trim().toLowerCase();
    const truthyValues = ['true', 'yes', '1'];
    const falsyValues = ['false', 'no', '0', '', 'null', 'undefined'];

    if (truthyValues.includes(normalizedStr)) {
        return true;
    } else if (falsyValues.includes(normalizedStr)) {
        return false;
    }

    return false;
}