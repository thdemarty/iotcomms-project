#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/_types.h>

#include "net/gnrc/nettype.h"
#include "net/gnrc/pktbuf.h"
#include "net/netif.h"
#include "net/netopt.h"
#include "ztimer.h"
#include "assert.h"
#include "net/gnrc.h"
#include "net/gnrc/ipv6.h"
#include "nimble_netif.h"
#include "net/gnrc/netif.h"
#include "nimble_netif_conn.h"
#include "nimble_addr.h"
#include "host/ble_hs.h"
#include "thread.h"
#include "msg.h"
#include "net/bluetil/ad.h"
#include "net/bluetil/addr.h"
#include "host/ble_gap.h"
#include "board.h"
#include "ws281x.h"
#include "periph/gpio.h"
#include "controller/ble_phy.h"

#define NEOPIXEL_PIN GPIO_PIN(0, 16)

#define MSG_QUEUE_SIZE 8
#define BRIGHTNESS 0.2f

// BLE connection parameters
#define DEFAULT_SCAN_DURATION_MS 500U
#define DEFAULT_CONN_TIMEOUT_MS 500U
#define DEFAULT_SCAN_ITVL_MS 100U
#define DEFAULT_CONN_ITVL_MS 15U
#define DEFAULT_ADV_ITVL_MS 30U

static char receive_thread_stack[THREAD_STACKSIZE_DEFAULT];
static msg_t msg_queue[MSG_QUEUE_SIZE];

static ble_addr_t peer_addr[] = {
    {.type = BLE_ADDR_RANDOM, .val = {0xc0, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc1, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc2, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc3, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
    {.type = BLE_ADDR_RANDOM, .val = {0xc4, 0xbb, 0xcc, 0xdd, 0xee, 0xff}},
};

static ws281x_pixel_t peer_colors[] = {
    {.r = 255 * BRIGHTNESS, .g = 0 * BRIGHTNESS, .b = 0 * BRIGHTNESS},   // node 0
    {.r = 0 * BRIGHTNESS, .g = 255 * BRIGHTNESS, .b = 0 * BRIGHTNESS},   // node 1
    {.r = 0 * BRIGHTNESS, .g = 0 * BRIGHTNESS, .b = 255 * BRIGHTNESS},   // node 2
    {.r = 255 * BRIGHTNESS, .g = 255 * BRIGHTNESS, .b = 0 * BRIGHTNESS}, // node 3
    {.r = 0 * BRIGHTNESS, .g = 255 * BRIGHTNESS, .b = 255 * BRIGHTNESS}, // node 4
};

static uint8_t led_buffer[1 * WS281X_BYTES_PER_DEVICE];

static char led_thread_stack[THREAD_STACKSIZE_DEFAULT];

void *led_status_thread(void *args)
{
    (void)args;

    while (1)
    {
        unsigned count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);

        if (count >= (NODE_COUNT - 1))
        {
            LED1_ON;
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
        else
        {
            LED1_TOGGLE;
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
    }
    return NULL;
}

static void advertise_to(ble_addr_t *ble_addr)
{
    int res;
    (void)res;

    nimble_netif_accept_cfg_t accept_cfg = {
        .own_addr_type = BLE_ADDR_RANDOM,
        .timeout_ms = 0,
    };

    // direct advertising (non-scannable)
    res = nimble_netif_accept_direct(ble_addr, &accept_cfg);

    if (res != 0)
    {
        printf("[BLE] Direct Advertising error: %d\n", res);
    }
}

static void event_cb(int handle, nimble_netif_event_t event,
                     const uint8_t *addr)
{
    (void)addr;
    switch (event)
    {
    case NIMBLE_NETIF_ACCEPTING:
        printf("[BLEE] Advertising\n");
        break;

    case NIMBLE_NETIF_ACCEPT_STOP:
        printf("[BLEE] Stop Advertising\n");
        break;

    case NIMBLE_NETIF_INIT_SLAVE:
        printf("[BLEE] Incoming connection attempt as slave\n");
        break;

    case NIMBLE_NETIF_CONNECTED_SLAVE:
        printf("[BLEE] Connected as slave, handle=%d addr=%02x:%02x:%02x:%02x:%02x:%02x\n", handle, addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
        break;

    case NIMBLE_NETIF_CLOSED_SLAVE:
        printf("[BLEE] Slave connection closed, handle=%d\n", handle);
        break;

    case NIMBLE_NETIF_INIT_MASTER:
        printf("[BLEE] Starting connection attempt as master\n");
        break;

    case NIMBLE_NETIF_CONNECTED_MASTER:
        printf("[BLEE] Connected as master, handle=%d addr=%02x:%02x:%02x:%02x:%02x:%02x\n", handle, addr[5], addr[4], addr[3], addr[2], addr[1], addr[0]);
        break;

    case NIMBLE_NETIF_CLOSED_MASTER:
        printf("[BLEE] Master connection closed, handle=%d\n", handle);
        break;

    default:
        break;
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

static void setup_ble_stack(void)
{
    nimble_netif_connect_cfg_t connect_cfg = {
        .scan_itvl_ms = 100,
        .scan_window_ms = 50,
        .conn_itvl_min_ms = DEFAULT_CONN_ITVL_MS,
        .conn_itvl_max_ms = DEFAULT_CONN_ITVL_MS,
        .conn_supervision_timeout_ms = 2000,
        .own_addr_type = BLE_ADDR_RANDOM,
        .timeout_ms = 2000,
    };

    for (int i = 0; i < NODEID; i++)
    {
        uint8_t reversed_addr[6];
        bluetil_addr_swapped_cp(peer_addr[i].val, reversed_addr);

        if (nimble_netif_conn_get_by_addr(reversed_addr) == NIMBLE_NETIF_CONN_INVALID)
        {
            printf("[BLECONN] Advertising to peer %d (not connected)\n", i);
            // clean up any previous advertising
            nimble_netif_accept_stop();
            // start advertising to this peer
            advertise_to(&peer_addr[i]);

            ztimer_sleep(ZTIMER_MSEC, 3000);

            nimble_netif_accept_stop();
            ztimer_sleep(ZTIMER_MSEC, 500);
        }
    }

    for (int i = NODEID + 1; i < NODE_COUNT; i++)
    {
        uint8_t reversed_addr[6];
        bluetil_addr_swapped_cp(peer_addr[i].val, reversed_addr);

        if (nimble_netif_conn_get_by_addr(reversed_addr) == NIMBLE_NETIF_CONN_INVALID)
        {
            printf("[BLECONN] Connecting to to peer %d (not connected)\n", i);

            int rc = nimble_netif_connect(&peer_addr[i], &connect_cfg);

            if (rc < 0)
            {
                printf("[BLECONN] Connection error: %d\n", rc);
            }

            ztimer_sleep(ZTIMER_MSEC, 3000);
        }
    }
}

int send_gnrc_packet(uint8_t *src_addr, gnrc_netif_t *netif, char *payload_str)
{
    gnrc_pktsnip_t *payload;
    gnrc_pktsnip_t *netif_hdr;
    gnrc_pktsnip_t *pkt;

    int CUSTOM_PROTO_TYPE = 253;

    payload = gnrc_pktbuf_add(NULL, payload_str, strlen(payload_str), CUSTOM_PROTO_TYPE);
    if (!payload)
    {
        printf("[GNRC] Failed to allocate payload\n");
        return 1;
    }

    netif_hdr = gnrc_netif_hdr_build(src_addr, BLE_ADDR_LEN, NULL, 0);
    if (!netif_hdr)
    {
        printf("[GNRC] Failed to allocate netif header\n");
        gnrc_pktbuf_release(payload);
        return 1;
    }
    gnrc_netif_hdr_set_netif(netif_hdr->data, netif);

    gnrc_netif_hdr_t *neth = (gnrc_netif_hdr_t *)netif_hdr->data;
    neth->flags |= GNRC_NETIF_HDR_FLAGS_BROADCAST;

    pkt = gnrc_pkt_prepend(payload, netif_hdr);
    if (!pkt)
    {
        printf("[GNRC] Failed to prepend netif header\n");
        gnrc_pktbuf_release(payload);
        return 1;
    }

    if (gnrc_netif_send(netif, pkt) <= 0)
    {
        printf("[GNRC] Failed to send gnrc packet\n");
        gnrc_pktbuf_release(pkt);
        return 1;
    }

    // printf("[GNRC] Packet sent\n");
    return 0;
}

// foreach connection callback
// int foreach_conn_callback(nimble_netif_conn_t *conn, int conn_handle, void *arg)
// {
//     (void)arg;

//     printf("[DEBUG] Connection handle: %d state=%d\n", conn_handle, conn->state);
//     return 0;
// }

void *gnrc_receive_handler(void *args)
{
    (void)args;

    msg_t msg;
    msg_init_queue(msg_queue, MSG_QUEUE_SIZE);

    struct gnrc_netreg_entry me_reg =
        GNRC_NETREG_ENTRY_INIT_PID(GNRC_NETREG_DEMUX_CTX_ALL, thread_getpid());
    // gnrc_netreg_register(GNRC_NETTYPE_UNDEF, &me_reg);
    // gnrc_netreg_register(GNRC_NETTYPE_NETIF, &me_reg);
    /*
    ** Ok, this is stupid.
    ** I did not get our custom protocol over BLE to work.
    ** But the receive somehow treats this as IPv6 packets.
    */
    gnrc_netreg_register(GNRC_NETTYPE_IPV6, &me_reg);
    // gnrc_netreg_register(GNRC_NETTYPE_L2_DISCOVERY, &me_reg);

    while (1)
    {
        msg_receive(&msg);
        if (msg.type == GNRC_NETAPI_MSG_TYPE_RCV)
        {
            gnrc_pktsnip_t *pkt = msg.content.ptr;
            uint32_t timer = ztimer_now(ZTIMER_MSEC);
            
            gnrc_pktsnip_t *netif_snip = gnrc_pktsnip_search_type(pkt, GNRC_NETTYPE_NETIF);
            
            if (netif_snip == NULL) {
                printf("[WARN] No NETIF header found in packet\n");
                gnrc_pktbuf_release(pkt);
                continue;
            }

            gnrc_netif_hdr_t *hdr = (gnrc_netif_hdr_t *)netif_snip->data;

            int rssi_raw = hdr->rssi;
            int lqi_raw = hdr->lqi;

            // for (size_t i = 0; i < pkt->next->size; i++) {
            //   printf(" %02x", ((uint8_t *)pkt->next->data)[i]);
            // }
            // printf("\n");

            size_t data_size = pkt->next->size;
            int node_id = -1;
            if (14 < data_size)
            {
                node_id = (int)((uint8_t *)pkt->next->data)[13] & 0x0F;
            }

            int rc = 0;
            int8_t rssi_gap = INT8_MAX;
            uint8_t ble_addr[6];
            for (int i = 0; i < 6; i++)
            {
                ble_addr[i] = ((uint8_t *)pkt->next->data)[8 + i];
            }

            int conn_handle = nimble_netif_conn_get_by_addr(ble_addr);

            nimble_netif_conn_t *conn;
            conn = nimble_netif_conn_get(conn_handle);
            rc = ble_gap_conn_rssi(conn->gaphandle, &rssi_gap);
            if (rc != 0)
            {
                printf("[WARN] error reading gap rssi: %d for handle: %d\n", rc, conn_handle);
            }

            if (rssi_raw == GNRC_NETIF_HDR_NO_RSSI)
            {
                rssi_raw = rssi_gap;
            }

            if (lqi_raw == GNRC_NETIF_HDR_NO_LQI)
            {
                lqi_raw = 0; // LQI not available, set to 0 or some default value
            }

            struct ble_gap_conn_desc desc;
            rc = ble_gap_conn_find(conn->gaphandle, &desc);
            if (rc != 0)
            {
                printf("[WARN] error reading gap connection: %d for handle: %d\n", rc, conn_handle);
            }

            uint16_t latency = desc.conn_latency;

            // printf("[DEBUG] payload as string: \"%d\"\n", node_id);

            // printf("[DEBUG] NODE: %d, RSSI: %d, Latency: %u\n", node_id, rssi_raw, latency);
            printf("[DATA] %d, %lu, %d, %u\n", node_id, timer, rssi_raw, latency);

            // fix memory leak here
            gnrc_pktbuf_release(pkt);
        }
    }
}

int main(void)
{
    // Delay generally required before pyterm comes up
    ztimer_sleep(ZTIMER_MSEC, 3000);

    printf("[DEBUG] NODEID is: %d\n", NODEID);

    // Set up RGB LED for NODEID
    ws281x_params_t params = {
        .buf = led_buffer,
        .pin = NEOPIXEL_PIN,
        .numof = 1,
    };

    ws281x_t dev;

    if (ws281x_init(&dev, &params) != 0)
    {
        printf("[ERROR] Failed to initialize ws281x device\n");
        return 1;
    }

    ws281x_set(&dev, 0, peer_colors[NODEID]);
    ws281x_write(&dev);

    while (!ble_hs_synced())
    {
        ztimer_sleep(ZTIMER_MSEC, 100);
    }

    thread_create(
        led_thread_stack,
        sizeof(led_thread_stack),
        THREAD_PRIORITY_MAIN - 2,
        THREAD_CREATE_NO_STACKTEST,
        led_status_thread,
        NULL,
        "led_thread");

    nimble_netif_eventcb(event_cb);

    // Set own static random address
    int rc = ble_hs_id_set_rnd(peer_addr[NODEID].val);
    assert(rc == 0);

    // print BLE MAC address
    uint8_t own_addr[6];
    ble_hs_id_copy_addr(BLE_ADDR_RANDOM, own_addr, NULL);
    printf("[DEBUG] Own BLE address: %02x:%02x:%02x:%02x:%02x:%02x\n", own_addr[5],
           own_addr[4], own_addr[3], own_addr[2], own_addr[1], own_addr[0]);

    ztimer_sleep(ZTIMER_MSEC, 200);

    setup_ble_stack();

    // Handle incoming messages in separate thread
    thread_create(
        receive_thread_stack,
        sizeof(receive_thread_stack),
        THREAD_PRIORITY_MAIN - 1,
        THREAD_CREATE_NO_STACKTEST,
        gnrc_receive_handler,
        NULL,
        "receive_thread");

    // Continuously send packets
    gnrc_netif_t *netif = find_ble_netif();
    char payload[8];
    sprintf(payload, "NODE_%d", NODEID);

    setup_ble_stack();

    while (1)
    {
        unsigned count = nimble_netif_conn_count(NIMBLE_NETIF_L2CAP_CONNECTED);

        // Send data only if we are connected to all other nodes, otherwise keep trying to connect
        if (count == (NODE_COUNT - 1))
        {
            send_gnrc_packet(own_addr, netif, payload);
            ztimer_sleep(ZTIMER_MSEC, 100);
        }
        else
        {
            printf("[DEBUG] Waiting for mesh to form... (%u/%d)\n", count, NODE_COUNT-1);
            setup_ble_stack();
        }
    }
}
