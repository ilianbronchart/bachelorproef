// This script is used to sort the "Created" column in the DataTable

function parseCustomDate(s) {
  // Expected format: "03/02/23 at 08:16 AM"
  var parts = s.match(/(\d{2})\/(\d{2})\/(\d{2})\s+at\s+(\d{2}):(\d{2})\s+(AM|PM)/);
  if (!parts) return new Date(0); // default fallback date

  var day = parseInt(parts[1], 10);
  var month = parseInt(parts[2], 10) - 1;  // JavaScript months are 0-indexed
  var year = parseInt(parts[3], 10) + 2000; // adjust as needed for 2-digit year

  var hour = parseInt(parts[4], 10);
  var minute = parseInt(parts[5], 10);
  var ampm = parts[6];

  if (ampm === 'PM' && hour < 12) hour += 12;
  if (ampm === 'AM' && hour === 12) hour = 0;

  return new Date(year, month, day, hour, minute);
}

jQuery.extend(jQuery.fn.dataTable.ext.type.order, {
  "custom-date-pre": function (d) {
    return parseCustomDate(d).getTime();
  }
});

function registerRecordingsTable(id) {
  $(id).DataTable({
    columnDefs: [
      { type: "custom-date", targets: 1 },
      { orderable: false, targets: 3 }
    ],
    order: [[1, 'desc']]
  });
}