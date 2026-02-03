#include <stdio.h>
#include "controller/ble_phy.h"
#include "host/ble_hs.h" // for BLE Host Stack
#include "host/ble_gap.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "ztimer.h"

// Maximum transmission power of +8 dBm (according to nRF52 datasheet)
// See: https://cdn-learn.adafruit.com/assets/assets/000/092/427/original/nRF52840_PS_v1.1.pdf

// Set transmission power to +8 dBm (maximum for nRF52)
// in dBm
#define TX_POWER 8

const char *device_name = "Node nRF52";
uint8_t addr_type;

void advertise(void)
{
	struct ble_gap_adv_params adv_params;
	struct ble_hs_adv_fields fields;
	int rc;

	/* Fill all fields and parameters with zeros */
	memset(&adv_params, 0, sizeof(adv_params));
	memset(&fields, 0, sizeof(fields));

	adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
	adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

	fields.flags = BLE_HS_ADV_F_DISC_GEN;
	fields.name = (uint8_t *)device_name;
	fields.name_len = strlen(device_name);
	fields.name_is_complete = 1;

	fields.tx_pwr_lvl = TX_POWER;
	fields.tx_pwr_lvl_is_present = 1;

	rc = ble_gap_adv_set_fields(&fields);
	assert(rc == 0);
	rc = ble_gap_adv_start(addr_type, NULL, 100,
						   &adv_params, NULL, NULL);
	assert(rc == 0);
}

int main(void)
{
    int rc;

	ztimer_sleep(ZTIMER_MSEC, 5000); // Wait for system to stabilize

	rc = ble_phy_txpwr_set(TX_POWER);
	assert(rc == 0);

	rc = ble_svc_gap_device_name_set(device_name);
    assert(rc == 0);

	rc = ble_hs_util_ensure_addr(0);
    assert(rc == 0);

	rc = ble_hs_id_infer_auto(0, &addr_type);
    assert(rc == 0);

	
    while (1) {
		advertise();
		printf("Advertising as %s\n", device_name);
        ztimer_sleep(ZTIMER_SEC, 1);
    }

    return 0;
}
