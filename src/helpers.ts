type AnyObject = { [key: string]: any };

function toCamelCaseKey(key: string): string {
  return key.replace(/_([a-z])/g, (_, char) => char.toUpperCase());
}

function toSnakeCaseKey(key: string): string {
  return key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
}

function convertKeys(obj: any, keyConverter: (key: string) => string, deep = false): any {
  if (Array.isArray(obj)) {
    return obj.map((item) =>
      deep && typeof item === "object" && item !== null
        ? convertKeys(item, keyConverter, deep)
        : item
    );
  }
  if (obj && typeof obj === "object" && obj.constructor === Object) {
    const newObj: AnyObject = {};
    for (const [key, value] of Object.entries(obj)) {
      const newKey = keyConverter(key);
      newObj[newKey] =
        deep && typeof value === "object" && value !== null
          ? convertKeys(value, keyConverter, deep)
          : value;
    }
    return newObj;
  }
  return obj;
}

/**
 * Converts the keys of an object from snake_case to camelCase.
 *
 * @param obj - The object whose keys should be converted.
 * @param deep - If true, keys in nested objects/arrays will also be converted.
 * @returns A new object with camelCase keys.
 */
export function objectToCamelCase(obj: Record<string, any>, deep = false) {
  return convertKeys(obj, toCamelCaseKey, deep);
}

/**
 * Converts the keys of an object from camelCase to snake_case.
 *
 * @param obj - The object whose keys should be converted.
 * @param deep - If true, keys in nested objects/arrays will also be converted.
 * @returns A new object with snake_case keys.
 */
export function objectToSnakeCase(obj: Record<string, any>, deep = false) {
  return convertKeys(obj, toSnakeCaseKey, deep);
}
