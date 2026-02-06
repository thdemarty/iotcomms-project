#include <stdint.h>
#include <stdio.h>

#include "net/gnrc/nettype.h"
#include "net/netopt.h"
#include "ztimer.h"
#include "assert.h"
#include "net/ipv6/addr.h"
#include "net/gnrc.h"
#include "nimble_netif.h"
#include "nimble_addr.h"
#include "host/ble_hs.h"
#include "thread.h"
#include "msg.h"

#define NODE_COUNT 5
#define MSG_QUEUE_SIZE 8

static const char *addr_node_str[] = {"2001:db8::1", "2001:db8::2", "2001:db8::3", "2001:db8::4", "2001:db8::5"};
static ipv6_addr_t addr_node[NODE_COUNT];
static char receive_thread_stack[THREAD_STACKSIZE_DEFAULT];
static msg_t msg_queue[MSG_QUEUE_SIZE];

// TODO: replace with real device MAC
// we want to have static random addresses with MSB starting with 0b11
// FIXME: i think this might by in little endian?
static ble_addr_t peer_addr[] = {
    {
        .type = BLE_ADDR_RANDOM,
        .val = {0xc0, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_RANDOM,
        .val = {0xc1, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_RANDOM,
        .val = {0xc2, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_RANDOM,
        .val = {0xc3, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
    {
        .type = BLE_ADDR_RANDOM,
        .val = {0xc4, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
    },
};

static gnrc_netif_t *ble_netif = NULL;

// Function to generate advertising packet
static void get_adv_packet(uint8_t *adv_data, size_t *adv_data_len)
{
    char device_name[20];
    snprintf(device_name, sizeof(device_name), "Node-%d", NODEID);
    size_t name_len = strlen(device_name);
    if (name_len + 2 > *adv_data_len)
    {
        printf("Advertising data buffer too small\n");
        *adv_data_len = 0;
        return;
    }

    adv_data[0] = name_len + 1;
    adv_data[1] = BLE_HS_ADV_TYPE_COMP_NAME;
    memcpy(&adv_data[2], device_name, name_len);

    *adv_data_len = name_len + 2;
}

static void event_cb(int handle, nimble_netif_event_t event,
                     const uint8_t *addr)
{
    (void)addr;
    switch (event)
    {
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
    if (res < 0)
    {
        printf("Failed to add IPv6 address to interface %u: %d\n",
               netif->pid, res);
    }
    else
    {
        char str[IPV6_ADDR_MAX_STR_LEN];
        printf("Added IPv6 address %s to interface %u\n",
               ipv6_addr_to_str(str, addr, sizeof(str)), netif->pid);
    }
}

static gnrc_netif_t *find_ble_netif(void)
{
    gnrc_netif_t *netif = NULL;
    while ((netif = gnrc_netif_iter(netif)))
    {
        if (netif->device_type == NETDEV_TYPE_BLE)
        {
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

// FIXME: We need to do anything related to the BLE stack in the ble_hs_sync 
// callback, which is called when the stack is ready. Normally we could set 
// it as a callback with ble_hs_cfg.sync_cb, but I could not get that to work,
// so I called the function manually in the main
static void ble_on_sync(void)
{
    // Set own static random address
    int rc = ble_hs_id_set_rnd(peer_addr[NODEID].val);

    // Start advertising
    uint8_t adv_data[31];
    size_t adv_data_len = sizeof(adv_data);
    get_adv_packet(adv_data, &adv_data_len);

    nimble_netif_accept_cfg_t accept_cfg = {
        .flags = NIMBLE_NETIF_FLAG_LEGACY,
        .own_addr_type = BLE_ADDR_RANDOM,
        .channel_map = 0,
        .tx_power = 8, // in dBm
        .adv_itvl_ms = 100,
        .timeout_ms = 0, // in ms
    };
    rc = nimble_netif_accept(adv_data, adv_data_len, &accept_cfg);
    if (rc != 0)
    {
        printf("nimble_netif_accept failed: %d\n", rc);
    }
    else
    {
        printf("Started advertising with data length %zu\n", adv_data_len);
    }

    // TODO: un-comment to connect to other nodes, was commented out for 
    // testing purposes.

    // Connect to other nodes
    // nimble_netif_connect_cfg_t connect_cfg = {0};
    // for (int n = 0; n < NODE_COUNT; n++) {
    //     if (n != NODEID) {
    //         // Note: nimble_netif_connect est non-bloquant
    //         nimble_netif_connect(&peer_addr[n], &connect_cfg);
    //     }
    // }
}

int send_gnrc_packet(ipv6_addr_t *dst_addr, gnrc_netif_t *netif)
{
    int rc;
    char *pld = "Payload";

    // FIXME: find out why this fails
    rc = gnrc_netapi_set(netif->pid, NETOPT_IPV6_ADDR, 0, dst_addr, sizeof(*dst_addr));
    if (rc != 0) {
        printf("Failed to set destination address: %d\n", rc);
    }

    int PROTO_TYPE = GNRC_NETTYPE_IPV6;
    gnrc_pktsnip_t *payload =
        gnrc_pktbuf_add(NULL, pld, strlen(pld), PROTO_TYPE);
    if (!payload) {
        printf("Failed to allocate payload\n");
        return 1;
    }
    gnrc_pktsnip_t *netif_hdr = gnrc_netif_hdr_build(NULL, 0, NULL, 0);
    if (!netif_hdr) {
        printf("Failed to allocate link-layer header\n");
        gnrc_pktbuf_release(payload);
        return 1;
    }

    // Set the network interface
    gnrc_netif_hdr_set_netif(netif_hdr->data, netif);

    gnrc_netif_hdr_t *neth = (gnrc_netif_hdr_t *)netif_hdr->data;
    // FIXME: Broadcast might fail silently
    neth->flags |= GNRC_NETIF_HDR_FLAGS_BROADCAST; 

    // Prepend header to payload
    gnrc_pktsnip_t *pkt = gnrc_pkt_prepend(payload, netif_hdr);

    // Send packet to network interface - GNRC will handle L2 forwarding
    if (!gnrc_netif_send(netif, pkt)) {
        printf("Failed to send packet\n");
        gnrc_pktbuf_release(pkt);
        return 1;
    }
    printf("Packet sent\n");

    return 0;
}

void *gnrc_receive_handler(void *args){
    (void) args;

    msg_t msg;
    msg_init_queue(msg_queue, MSG_QUEUE_SIZE);

    struct gnrc_netreg_entry me_reg =
        GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me_reg);

    while (1) {
        msg_receive(&msg);
        if (msg.type == GNRC_NETAPI_MSG_TYPE_RCV) {
            printf("RCV: 4\n");
            gnrc_pktsnip_t *pkt = msg.content.ptr;
            if (pkt->next) {
              if (pkt->next->next) {
                gnrc_netif_hdr_t *hdr = pkt->next->next->data;
                int rssi_raw = (int)hdr->rssi;
                int lqi_raw = (int)hdr->lqi;
                printf("RSSI: %d, LQI: %d", rssi_raw, lqi_raw);
              }
            }
        }
    }
}

int main(void) {
    // Delay generally required before pyterm comes up
    ztimer_sleep(ZTIMER_MSEC, 3000);

    printf("NODEID is: %d\n", NODEID);

    while (!ble_hs_synced()) {
        ztimer_sleep(ZTIMER_MSEC, 100);
    }

    nimble_netif_eventcb(event_cb);

    ble_on_sync();

    // print BLE MAC address
    uint8_t own_addr[6];
    ble_hs_id_copy_addr(BLE_ADDR_RANDOM, own_addr, NULL);
    printf("Own BLE address: %02x:%02x:%02x:%02x:%02x:%02x\n", own_addr[5],
           own_addr[4], own_addr[3], own_addr[2], own_addr[1], own_addr[0]);

    printf("Got beyond nimble!\n");

    // TODO: find out whether we can get away with raw GNRC without IPv6
    for (int n = 0; n < NODE_COUNT; n++) {
        get_node_addr(n);
    }
    
    // Assign IPv6 address to own BLE interface
    // Might need to wait for the BLE interface to come up
    gnrc_netif_t *netif = find_ble_netif();
    if (netif != NULL && netif != ble_netif) {
        ble_netif = netif;
        const ipv6_addr_t *my_addr = &addr_node[NODEID];
        assign_static_ipv6(ble_netif, my_addr);
    } else {
        printf("Error: no BLE interface\n");
    }

    // Handle incoming messages in separate thread
    thread_create(
        receive_thread_stack,
        sizeof(receive_thread_stack),
        THREAD_PRIORITY_MAIN - 1,
        THREAD_CREATE_NO_STACKTEST,
        gnrc_receive_handler,
        NULL,
        "receive_thread"
    );

    // Continuously send packets
    for (int n = 0; n < NODE_COUNT; n++) {
        if (n != NODEID) {
            send_gnrc_packet(&addr_node[n], netif);
        }
    }
}
