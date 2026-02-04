#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ztimer.h"
#include "assert.h"
#include "net/ipv6/addr.h"
#include "net/gnrc/netif.h"
#include "net/gnrc/netapi.h"
#include "nimble_netif.h"
#include "nimble_addr.h"
#include "host/ble_hs.h"


#define NODE_COUNT 5
static const char *addr_node_str[] = {"2001:db8::1", "2001:db8::2", "2001:db8::3", "2001:db8::4", "2001:db8::5"}; 
static ipv6_addr_t addr_node[NODE_COUNT];

// TODO: replace with real device MAC
static ble_addr_t peer_addr[] = {
    {
        .type = BLE_ADDR_PUBLIC,
        .val = {0xa0, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_PUBLIC,
        .val = {0xa1, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_PUBLIC,
        .val = {0xa2, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_PUBLIC,
        .val = {0xa3, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_PUBLIC,
        .val = {0xa4, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
};

static gnrc_netif_t *ble_netif = NULL;

static void event_cb(int handle, nimble_netif_event_t event,
                      const uint8_t *addr)
{
    (void) addr;
    switch (event) {
        case NIMBLE_NETIF_ACCEPTING:
            printf("Advertising\n");
            break;

        case NIMBLE_NETIF_CONNECTED_SLAVE:
            printf("Connected as slave, handle=%d\n", handle);
            break;

        case NIMBLE_NETIF_CLOSED_SLAVE:
            printf("Slave connection closed, handle=%d\n", handle);
            break;

        case NIMBLE_NETIF_INIT_MASTER:
            puts("Starting connection attempt");
            break;

        case NIMBLE_NETIF_CONNECTED_MASTER:
            printf("Connected as master, handle=%d\n", handle);
            break;

        case NIMBLE_NETIF_CLOSED_MASTER:
            printf("Master connection closed, handle=%d\n", handle);
            break;

        default:
            break;
    }
}

static void assign_static_ipv6(gnrc_netif_t *netif, const ipv6_addr_t *addr)
{
    uint8_t flags = GNRC_NETIF_IPV6_ADDRS_FLAGS_STATE_VALID;
    int res = gnrc_netif_ipv6_addr_add(netif, addr, 64, flags);
    if (res < 0) {
        printf("Failed to add IPv6 address to interface %u: %d\n",
               netif->pid, res);
    } else {
        char str[IPV6_ADDR_MAX_STR_LEN];
        printf("Added IPv6 address %s to interface %u\n",
               ipv6_addr_to_str(str, addr, sizeof(str)), netif->pid);
    }
}

static gnrc_netif_t *find_ble_netif(void)
{
    gnrc_netif_t *netif = NULL;
    while ((netif = gnrc_netif_iter(netif))) {
        if (netif->device_type == NETDEV_TYPE_BLE) {
            return netif;
        }
    }
    return NULL;
}

ipv6_addr_t *get_node_addr(uint8_t node_id)
{
    ipv6_addr_t *rc;
    rc = ipv6_addr_from_str(&addr_node[node_id], addr_node_str[node_id]);
    assert(rc != NULL);
    return &addr_node[node_id];
}

int main(void)
{
    // Delay generally required before pyterm comes up 
    ztimer_sleep(ZTIMER_MSEC, 3000);

    printf("NODEID is: %d\n",NODEID);

    // TODO: find out why this is failing
    nimble_netif_init();
    printf("1\n");

    // Print BLE address of this node
    uint8_t own_addr[6];
    int rc = ble_hs_id_copy_addr(BLE_ADDR_PUBLIC, own_addr, NULL);
    if (rc != 0) {
        printf("Failed to get own BLE address: %d",rc);
    } else {
        printf("BLE addr: %02X:%02X:%02X:%02X:%02X:%02X\n",
               own_addr[5], own_addr[4], own_addr[3], own_addr[2], own_addr[1], own_addr[0]);
    }

    printf("2\n");

    // TODO: Find correct flags for our use case
    nimble_netif_accept_cfg_t accept_cfg = {.flags = NIMBLE_NETIF_FLAG_LEGACY};
    nimble_netif_connect_cfg_t connect_cfg = {0};

    printf("3\n");
    nimble_netif_eventcb(event_cb);
    printf("4\n");

    // TODO: find out why this is failing
    nimble_netif_accept(NULL, 0, &accept_cfg);
    printf("5\n");

    for(int n = 0; n < NODE_COUNT; n++) {
        if (n != NODEID) {
            int res = nimble_netif_connect(&peer_addr[n], &connect_cfg);
            if (res != 0) {
                printf("nimble_netif_connect failed: %d\n", res);
            }
        }
    }

    printf("Got beyond nimble!\n");
    // Assign IPv6 address to own BLE interface
    // Might need to wait for the BLE interface to come up
    while (1) {
        ztimer_sleep(ZTIMER_MSEC, 100);

        gnrc_netif_t *netif = find_ble_netif();
        if (netif != NULL && netif != ble_netif) {
            ble_netif = netif;
            const ipv6_addr_t *my_addr = get_node_addr(NODEID);
            assign_static_ipv6(ble_netif, my_addr);
        }
    }

    // TODO: send pings (there is a module for that), do measurements

    return 0;
}
